use ipc_channel::ipc;
use nix::unistd::{fork, ForkResult};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFunction, PyModule, PyTuple};
use pyo3::Python;
use std::process;
use std::sync::{Arc, Mutex};
use log::{info, warn, error, trace, debug};

#[derive(Clone)]
struct FunctionInfo {
    module_name: String,
    function_name: String,
    pickled_func: Vec<u8>,
}

// Parent's view of IPC channels for one replica
struct ParentEndChannels {
    tx_to_child: ipc::IpcSender<Vec<u8>>,   // Parent sends arguments to child
    rx_from_child: ipc::IpcReceiver<Vec<u8>>, // Parent receives results from child
}

// Information about a single replica process
struct ReplicaProcessInfo {
    ipc_channels: ParentEndChannels,
    child_pid: nix::unistd::Pid,
}

#[pyclass()]
struct AsyncPool {
    child_processes: Vec<Arc<Mutex<ReplicaProcessInfo>>>,
    function_info: FunctionInfo, // Cloned into each child process
    next_replica_idx: Arc<Mutex<usize>>, // For round-robin load balancing
}

fn child_loop(
    function_info: FunctionInfo,
    rx: ipc::IpcReceiver<Vec<u8>>,
    tx: ipc::IpcSender<Vec<u8>>,
) {
    info!("[CHILD] Starting child loop with PID: {}", process::id());
    // Signal parent that we have initialized.
    if let Err(e) = tx.send(vec![]) {
        error!("[CHILD] Failed to send initialization signal: {:?}", e);
    }

    Python::with_gil(|py| {
        info!("[CHILD] Unpickling function with cloudpickle");
        let cloudpickle = match py.import("cloudpickle") {
            Ok(m) => m,
            Err(e) => {
                error!("[CHILD] Error importing cloudpickle: {:?}", e);
                return;
            }
        };
        let asyncio = py.import("asyncio").unwrap();
        let event_loop = asyncio.call_method0("new_event_loop").unwrap();
        let inspect = py.import("inspect").unwrap();
        let dumps = cloudpickle.getattr("dumps").unwrap();
        let loads = cloudpickle.getattr("loads").unwrap();

        let func = match || -> PyResult<Bound<'_, PyFunction>> {
            let unpickled = loads.call1((PyBytes::new(py, &function_info.pickled_func),))?;
            let func = unpickled.downcast::<PyFunction>()?;
            Ok(func.clone())
        }() {
            Ok(f) => f,
            Err(e) => {
                error!("[CHILD] Failed to unpickle function: {:?}", e);
                return;
            }
        };
        // check if the function is a coroutine
        let is_coroutine = inspect
            .getattr("iscoroutinefunction")
            .unwrap()
            .call1((func.clone(),))
            .unwrap()
            .extract::<bool>()
            .unwrap();
        info!(
            "[CHILD] Function unpickled successfully: {}",
            function_info.function_name
        );

        loop {
            debug!("[CHILD] Waiting for pickled arguments from parent...");
            let pickled_args = match rx.recv() {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("[CHILD] Error receiving arguments: {:?}, exiting", e);
                    break;
                }
            };
            // Unpickle the args to a Python object (expecting a tuple)
            let args_obj = match loads.call1((PyBytes::new(py, &pickled_args),)) {
                Ok(o) => o,
                Err(e) => {
                    error!("[CHILD] Error unpickling args: {:?}", e);
                    continue;
                }
            };
            let args_tuple = match args_obj.downcast::<PyTuple>() {
                Ok(t) => t,
                Err(e) => {
                    error!("[CHILD] Expected tuple of arguments, got error: {:?}", e);
                    continue;
                }
            };
            trace!("[CHILD] Calling function with unpickled arguments");
            let call_result_obj = match func.call(args_tuple, None) {
                Ok(r) => r,
                Err(e) => {
                    error!("[CHILD] Error calling function: {:?}", e);
                    continue;
                }
            };

            // Check if the result is awaitable (a coroutine) and run it
            let final_ret_obj = if is_coroutine {
                debug!("[CHILD] Function is a coroutine, awaiting result");
                let await_result = event_loop
                    .call_method1("run_until_complete", (call_result_obj,))
                    .unwrap();
                await_result
            } else {
                call_result_obj
            };
            // Pickle the return value and send it over IPC.
            let pickled_ret: Vec<u8> = match dumps.call1((final_ret_obj,)) {
                Ok(py_bytes) => match py_bytes.extract() {
                    Ok(b) => b,
                    Err(e) => {
                        error!("[CHILD] Error extracting pickled return: {:?}", e);
                        // TODO: Consider sending a pickled error back to the parent
                        continue;
                    }
                },
                Err(e) => {
                    error!("[CHILD] Error pickling return value: {:?}", e);
                    // TODO: Consider sending a pickled error back to the parent
                    continue;
                }
            };
            if let Err(e) = tx.send(pickled_ret) {
                error!("[CHILD] Failed to send return value: {:?}", e);
            }
        } // loop
        info!("[CHILD] Exiting child loop");
    });
}

#[pymethods]
impl AsyncPool {
    #[staticmethod]
    #[pyo3(signature=(func, *, replicas = 8))]
    fn wraps(py: Python<'_>, func: PyObject, replicas: usize) -> PyResult<Self> {
        if replicas == 0 {
            return Err(PyValueError::new_err("Number of replicas must be at least 1"));
        }
        info!("[PARENT] Creating AsyncPool from signature with {} replicas", replicas);

        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        let module_name = func_ref.getattr("__module__")?.extract::<String>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        info!(
            "[PARENT] Function identified: {}.{}",
            module_name, function_name
        );
        let cloudpickle = py.import("cloudpickle")?;
        let pickled_func: Vec<u8> = cloudpickle
            .getattr("dumps")?
            .call1((func.clone_ref(py),))?
            .extract()?;
        let function_info = FunctionInfo {
            module_name,
            function_name,
            pickled_func,
        };

        let mut all_replica_infos = Vec::with_capacity(replicas);

        for i in 0..replicas {
            debug!("[PARENT] Setting up IPC channels for replica {}", i + 1);
            // tx_args_p: parent sends arguments, rx_args_c: child receives arguments
            let (tx_args_p, rx_args_c): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
                ipc::channel().map_err(|e| PyValueError::new_err(format!("Failed to create args IPC channel: {}", e)))?;
            // tx_res_c: child sends results, rx_res_p: parent receives results (and init signal)
            let (tx_res_c, rx_res_p): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
                ipc::channel().map_err(|e| PyValueError::new_err(format!("Failed to create results IPC channel: {}", e)))?;

            info!("[PARENT] Forking child process {}/{}", i + 1, replicas);
            match unsafe { fork() } {
                Ok(ForkResult::Parent { child }) => {
                    info!("[PARENT] Forked child process {}/{} with PID: {}", i + 1, replicas, child);
                    // Wait for the child's initialization signal on its result channel.
                    match rx_res_p.recv() {
                        Ok(_) => info!("[PARENT] Child process {} ({}) initialized successfully", i + 1, child),
                        Err(e) => {
                            error!("[PARENT] Child process {} ({}) failed to initialize: {:?}", i + 1, child, e);
                            // TODO: Terminate already started children before erroring out.
                            // For now, if one child fails, the whole setup fails.
                            return Err(PyValueError::new_err(format!(
                                "Child process {} ({}) failed to initialize: {:?}",
                                i + 1, child, e
                            )));
                        }
                    }
                    let parent_channels = ParentEndChannels { tx_to_child: tx_args_p, rx_from_child: rx_res_p };
                    let replica_info = ReplicaProcessInfo { ipc_channels: parent_channels, child_pid: child };
                    all_replica_infos.push(Arc::new(Mutex::new(replica_info)));
                }
                Ok(ForkResult::Child) => {
                    info!("[CHILD] Child process forked with PID: {} (replica {}/{})", process::id(), i + 1, replicas);
                    unsafe { ffi::PyOS_AfterFork_Child() };
                    // Child uses rx_args_c to receive arguments and tx_res_c to send results (and init signal)
                    child_loop(function_info.clone(), rx_args_c, tx_res_c);
                    process::exit(0);
                }
                Err(e) => {
                    error!("[PARENT] Fork failed for replica {}: {}", i + 1, e);
                    // TODO: Terminate already started children.
                    return Err(PyValueError::new_err(format!("Fork failed for replica {}: {}", i + 1, e)));
                }
            }
        }

        info!("[PARENT] All {} replica processes setup complete", replicas);
        Ok(AsyncPool {
            child_processes: all_replica_infos,
            function_info, // Original function_info stored, cloned version passed to children
            next_replica_idx: Arc::new(Mutex::new(0)),
        })
    }

    #[pyo3(signature=(*args,))]
    fn __call__<'p>(
        &mut self,
        py: Python<'p>,
        args: Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'p, PyAny>> {
        if self.child_processes.is_empty() {
            error!("[PARENT] __call__ invoked on a AsyncPool with no replicas.");
            return Err(PyValueError::new_err("No replicas available to handle the call."));
        }
        
        let replica_arc = {
            let mut idx_guard = self.next_replica_idx.lock().unwrap_or_else(|e| {
                error!("[PARENT] Failed to lock next_replica_idx: {}", e);
                // Fallback or panic, here we panic as it's a critical state.
                // In a real scenario, might try to recover or return a specific error.
                panic!("Failed to lock next_replica_idx: {}", e);
            });
            let current_idx = *idx_guard;
            *idx_guard = (current_idx + 1) % self.child_processes.len();
            debug!("[PARENT] Selected replica {} for call", current_idx);
            self.child_processes[current_idx].clone()
        };

        let pickled_args: Vec<u8> = {
            let cloudpickle = py.import("cloudpickle")?;
            let dumps = cloudpickle.getattr("dumps")?;
            let py_bytes = dumps.call1((args,))?;
            py_bytes.extract()?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ipc_task_result = tokio::task::spawn_blocking(move || {
                // Lock the specific replica's info
                let guard = replica_arc.lock().map_err(|e| {
                    let msg = format!("Failed to acquire replica channel lock: {}", e);
                    error!("[PARENT] {}", msg);
                    msg 
                })?;
                
                trace!("[PARENT] Sending pickled arguments to child PID {}", guard.child_pid);
                guard.ipc_channels.tx_to_child.send(pickled_args).map_err(|e| {
                    let msg = format!("Failed to send arguments to child {}: {}", guard.child_pid, e);
                    error!("[PARENT] {}", msg);
                    msg
                })?;
                
                debug!("[PARENT] Waiting for pickled return value from child PID {}", guard.child_pid);
                guard.ipc_channels.rx_from_child.recv().map_err(|e| {
                    let msg = format!("Failed to receive data from child {}: {}", guard.child_pid, e);
                    error!("[PARENT] {}", msg);
                    msg
                })
            }).await;

            match ipc_task_result {
                Ok(Ok(pickled_ret)) => { // IPC successful, inner result is Ok(Vec<u8>)
                    // Unpickle the return value. This requires the GIL.
                    Python::with_gil(|py_inner| {
                        info!("[PARENT] IPC successful, unpickling result");
                        let cloudpickle = py_inner.import("cloudpickle")?;
                        let loads = cloudpickle.getattr("loads")?;
                        let py_bytes_ret = PyBytes::new_bound(py_inner, &pickled_ret);
                        let bound_obj = loads.call1((py_bytes_ret,))?;
                        Ok(bound_obj.to_object(py_inner)) // Convert to PyObject
                    })
                }
                Ok(Err(ipc_err_str)) => { // IPC error (String) from the blocking task's Result
                    // Error was already logged when ipc_err_str was created.
                    Err(PyValueError::new_err(ipc_err_str))
                }
                Err(join_err) => { // Task join error (e.g., panic in spawn_blocking)
                    let err_msg = format!("IPC task panicked or was cancelled: {}", join_err);
                    error!("[PARENT] {}", err_msg);
                    Err(PyValueError::new_err(err_msg))
                }
            }
        })
    }

    fn cleanup(&mut self) -> PyResult<()> {
        info!("[PARENT] Cleanup called, terminating {} child process(es)", self.child_processes.len());
        
        for (i, replica_arc) in self.child_processes.iter().enumerate() {
            match replica_arc.try_lock() { // Use try_lock to avoid blocking if a child is stuck holding the lock (unlikely here)
                Ok(replica_info) => {
                    info!("[PARENT] Terminating child process {} (PID: {})", i, replica_info.child_pid);
                    match nix::sys::signal::kill(replica_info.child_pid, nix::sys::signal::Signal::SIGTERM) {
                        Ok(_) => info!("[PARENT] Sent SIGTERM to child process PID {}", replica_info.child_pid),
                        Err(e) => warn!("[PARENT] Failed to send SIGTERM to child PID {}: {:?}", replica_info.child_pid, e),
                    }
                }
                Err(e) => {
                    // If we can't lock, we might not have the PID. This case should be rare.
                    // The PID is not mutable after creation, so lock is mainly for channel access.
                    // However, to be safe, we only kill if lock is acquired.
                    // Alternatively, store PIDs separately if cleanup needs to happen without lock.
                    // For now, log and continue.
                    warn!("[PARENT] Could not lock replica info for child {} during cleanup: {}. Skipping termination signal for this replica.", i, e);
                }
            }
        }
        // Clear the vector, which will drop the Arcs. If these are the last Arcs,
        // the Mutex<ReplicaProcessInfo> and then ReplicaProcessInfo (and its channels) will be dropped.
        self.child_processes.clear();
        info!("[PARENT] Child processes cleanup attempt finished.");
        Ok(())
    }
}

// Add Drop implementation for AsyncPool
impl Drop for AsyncPool {
    fn drop(&mut self) {
        info!("[PARENT] AsyncPool is being dropped, cleaning up");
        let _ = self.cleanup();
    }
}

#[pymodule]
fn gilboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the logger. You can customize this further if needed.
    // For example, to set a default log level if RUST_LOG is not set:
    // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Or simply:
    let _ = env_logger::try_init(); // Use try_init to avoid panic if already initialized
    info!("[MODULE] Initializing other_gil module");
    m.add_class::<AsyncPool>()?;
    Ok(())
}
