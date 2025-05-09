use ipc_channel::ipc;
use nix::unistd::{/*fork, ForkResult*/}; // ForkResult no longer needed, fork will be removed
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFunction, PyModule, PyTuple};
use pyo3::Python;
use std::process::{self, Command, Stdio}; // Added Command, Stdio
use std::env; // Added env
use std::sync::{Arc, Mutex};
use log::{info, warn, error, trace, debug};
use serde::{Serialize, Deserialize}; 
use std::os::unix::io::AsRawFd;
use nix::fcntl::{fcntl, FcntlArg, FdFlag};

#[derive(Clone, Serialize, Deserialize)] 
struct FunctionInfo {
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
    child_pid: nix::unistd::Pid, // Stays as nix::unistd::Pid, will convert from u32
}

#[pyclass()]
struct AsyncPool {
    child_processes: Vec<Arc<Mutex<ReplicaProcessInfo>>>,
    next_replica_idx: Arc<Mutex<usize>>, 
}

fn make_inheritable_receiver<T>(_: &ipc::IpcReceiver<T>) -> Result<(), nix::Error> {
    Ok(())
}

fn make_inheritable_sender<T>(_: &ipc::IpcSender<T>) -> Result<(), nix::Error> {
    Ok(())
}

// This child_loop is from the previous step, expecting FunctionInfo via IPC.
// It's called by child_entry_point in the spawned process.
fn child_loop(
    rx_from_parent: ipc::IpcReceiver<Vec<u8>>, 
    tx_to_parent: ipc::IpcSender<Vec<u8>>,   
) {
    let child_pid = process::id();
    info!("[CHILD PID {}] Starting child loop", child_pid);

    let function_info: FunctionInfo = match rx_from_parent.recv() {
        Ok(bytes) => match bincode::deserialize(&bytes) {
            Ok(fi) => fi,
            Err(e) => {
                error!("[CHILD PID {}] Failed to deserialize FunctionInfo: {:?}. Exiting.", child_pid, e);
                let _ = tx_to_parent.send(bincode::serialize(&Result::<(), String>::Err(format!("Deserialize FunctionInfo error: {}", e))).unwrap_or_default());
                process::exit(1);
            }
        },
        Err(e) => {
            error!("[CHILD PID {}] Failed to receive FunctionInfo: {:?}. Exiting.", child_pid, e);
            process::exit(1);
        }
    };
    info!("[CHILD PID {}] Received and deserialized FunctionInfo for: {}", child_pid, function_info.function_name);

    if let Err(e) = tx_to_parent.send(vec![]) { 
        error!("[CHILD PID {}] Failed to send initialization signal: {:?}", child_pid, e);
        process::exit(1);
    }
    info!("[CHILD PID {}] Sent initialization signal to parent.", child_pid);

    Python::with_gil(|py| {
        info!("[CHILD PID {}] Unpickling function '{}' with cloudpickle", child_pid, function_info.function_name);
        let cloudpickle = match py.import("cloudpickle") {
            Ok(m) => m,
            Err(e) => {
                error!("[CHILD PID {}] Error importing cloudpickle: {:?}", child_pid, e);
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
                error!("[CHILD PID {}] Failed to unpickle function: {:?}", child_pid, e);
                return;
            }
        };
        let is_coroutine = inspect
            .getattr("iscoroutinefunction")
            .unwrap()
            .call1((func.clone(),))
            .unwrap()
            .extract::<bool>()
            .unwrap();
        info!(
            "[CHILD PID {}] Function unpickled successfully: {}",
            child_pid, function_info.function_name
        );

        loop {
            debug!("[CHILD PID {}] Waiting for pickled arguments from parent...", child_pid);
            let pickled_args = match rx_from_parent.recv() {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("[CHILD PID {}] Error receiving arguments: {:?}, exiting", child_pid, e);
                    break;
                }
            };
            let args_obj = match loads.call1((PyBytes::new(py, &pickled_args),)) {
                Ok(o) => o,
                Err(e) => {
                    error!("[CHILD PID {}] Error unpickling args: {:?}", child_pid, e);
                    continue;
                }
            };
            let args_tuple = match args_obj.downcast::<PyTuple>() {
                Ok(t) => t,
                Err(e) => {
                    error!("[CHILD PID {}] Expected tuple of arguments, got error: {:?}", child_pid, e);
                    continue;
                }
            };
            trace!("[CHILD PID {}] Calling function with unpickled arguments", child_pid);
            let call_result_obj = match func.call(args_tuple, None) {
                Ok(r) => r,
                Err(e) => {
                    error!("[CHILD PID {}] Error calling function: {:?}", child_pid, e);
                    continue;
                }
            };

            let final_ret_obj = if is_coroutine {
                debug!("[CHILD PID {}] Function is a coroutine, awaiting result", child_pid);
                let await_result = event_loop
                    .call_method1("run_until_complete", (call_result_obj,))
                    .unwrap();
                await_result
            } else {
                call_result_obj
            };
            let pickled_ret: Vec<u8> = match dumps.call1((final_ret_obj,)) {
                Ok(py_bytes) => match py_bytes.extract() {
                    Ok(b) => b,
                    Err(e) => {
                        error!("[CHILD PID {}] Error extracting pickled return: {:?}", child_pid, e);
                        continue;
                    }
                },
                Err(e) => {
                    error!("[CHILD PID {}] Error pickling return value: {:?}", child_pid, e);
                    continue;
                }
            };
            if let Err(e) = tx_to_parent.send(pickled_ret) {
                error!("[CHILD PID {}] Failed to send return value: {:?}", child_pid, e);
            }
        } 
        info!("[CHILD PID {}] Exiting child loop", child_pid);
    });
    process::exit(0); // Ensure child process exits after loop
}

// New function to be called by child process logic from #[pymodule]
fn child_process_entry_point(
    rx_from_parent_b64: String, 
    tx_to_parent_b64: String
) -> PyResult<()> {
    let rx_bytes = base64::decode(&rx_from_parent_b64)
        .map_err(|e| PyValueError::new_err(format!("Child: Failed to decode rx_channel: {}", e)))?;
    let tx_bytes = base64::decode(&tx_to_parent_b64)
        .map_err(|e| PyValueError::new_err(format!("Child: Failed to decode tx_channel: {}", e)))?;

    let rx_from_parent: ipc::IpcReceiver<Vec<u8>> = bincode::deserialize(&rx_bytes)
        .map_err(|e| PyValueError::new_err(format!("Child: Failed to deserialize rx_channel: {}", e)))?;
    let tx_to_parent: ipc::IpcSender<Vec<u8>> = bincode::deserialize(&tx_bytes)
        .map_err(|e| PyValueError::new_err(format!("Child: Failed to deserialize tx_channel: {}", e)))?;
    
    pyo3::prepare_freethreaded_python(); 
    
    info!("[CHILD_SPAWN_ENTRY] Process detected as child worker. PID: {}. Calling child_loop.", std::process::id());
    
    child_loop(rx_from_parent, tx_to_parent);
    
    Ok(()) 
}

#[pyfunction]
pub fn child() -> PyResult<()> {
    // Initialize logging in child process context
    // let _ = env_logger::Builder::from_env(
    //     env_logger::Env::default().default_filter_or("info")
    // ).try_init();
    info!("[GILBOOST_CHILD] Initializing child process logging.");

    // Retrieve IPC channel information from environment
    let rx_b64 = std::env::var("GILBOOST_RX_CHANNEL_B64")
        .map_err(|e| PyValueError::new_err(format!("Child: Missing GILBOOST_RX_CHANNEL_B64: {}", e)))?;
    let tx_b64 = std::env::var("GILBOOST_TX_CHANNEL_B64")
        .map_err(|e| PyValueError::new_err(format!("Child: Missing GILBOOST_TX_CHANNEL_B64: {}", e)))?;

    info!("[GILBOOST_CHILD] Detected child process environment variables. PID: {}", std::process::id());
    info!("[GILBOOST_CHILD] RX channel: {}", rx_b64);
    info!("[GILBOOST_CHILD] TX channel: {}", tx_b64);

    // Launch the child process loop
    child_process_entry_point(rx_b64, tx_b64)
}

#[pymethods]
impl AsyncPool {
    #[staticmethod]
    #[pyo3(signature=(func, *, replicas = 8))]
    fn wraps(py: Python<'_>, func: PyObject, replicas: usize) -> PyResult<Self> {
        if replicas == 0 {
            return Err(PyValueError::new_err("Number of replicas must be at least 1"));
        }
        let parent_pid = process::id();
        info!("[PARENT PID {}] Creating AsyncPool with {} replicas using spawn.", parent_pid, replicas);

        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        let cloudpickle = py.import("cloudpickle")?;
        let pickled_func: Vec<u8> = cloudpickle
            .getattr("dumps")?
            .call1((func.clone_ref(py),))?
            .extract()?;
        let function_info = FunctionInfo {
            function_name,
            pickled_func,
        };

        let mut all_replica_infos = Vec::with_capacity(replicas);
        let current_python_exe = env::current_exe()
            .map_err(|e| PyValueError::new_err(format!("Failed to get current executable (python) path: {}", e)))?;
        
        info!("current python_exe: {:?}", current_python_exe);

        for i in 0..replicas {
            debug!("[PARENT PID {}] Setting up IPC channels for replica {}", parent_pid, i + 1);
            let (tx_to_child_p, rx_for_child_c): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
                ipc::channel().map_err(|e| PyValueError::new_err(format!("Failed to create to-child IPC channel: {}", e)))?;
            let (tx_from_child_c, rx_from_child_p): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
                ipc::channel().map_err(|e| PyValueError::new_err(format!("Failed to create from-child IPC channel: {}", e)))?;
            
            // Make the channels that are passed to the child inheritable.
            make_inheritable_receiver(&rx_for_child_c).map_err(|e| PyValueError::new_err(format!("Failed to set inheritable: {}", e)))?;
            make_inheritable_sender(&tx_from_child_c).map_err(|e| PyValueError::new_err(format!("Failed to set inheritable: {}", e)))?;

            let function_info_clone_for_child = function_info.clone();

            let rx_for_child_serialized = bincode::serialize(&rx_for_child_c)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize rx_for_child_c: {}", e)))?;
            let tx_from_child_serialized = bincode::serialize(&tx_from_child_c)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize tx_from_child_c: {}", e)))?;

            let rx_b64 = base64::encode(&rx_for_child_serialized);
            let tx_b64 = base64::encode(&tx_from_child_serialized);

            info!("[PARENT PID {}] Spawning child process {}/{}", parent_pid, i + 1, replicas);
            
            let mut cmd = Command::new(&current_python_exe);
            // cmd.arg("-c")
            //    .arg(format!("import {}; import sys; sys.exit(0)", this_module_name)) 
            //    .env("GILBOOST_CHILD_PROCESS", "1")
            //    .env("GILBOOST_RX_CHANNEL_B64", rx_b64)
            //    .env("GILBOOST_TX_CHANNEL_B64", tx_b64)
            //    .stdin(Stdio::null()); 
            cmd.arg("-c")
                .arg(format!("print('call gilboost');import gilboost; print('call child'); gilboost.child(); print('exit child'); import sys; sys.exit(0)"))
                .env("GILBOOST_CHILD_PROCESS", "1")
                .env("GILBOOST_RX_CHANNEL_B64", rx_b64)
                .env("GILBOOST_TX_CHANNEL_B64", tx_b64)
                .stdin(Stdio::null()); 
                

            if let Ok(rust_log_val) = env::var("RUST_LOG") {
                cmd.env("RUST_LOG", rust_log_val);
            }

            let mut child_process = cmd.spawn().map_err(|e| {
                PyValueError::new_err(format!("Failed to spawn child process {}/{}: {}", i + 1, replicas, e))
            })?;
            
            let child_id_u32 = child_process.id();
            info!("[PARENT PID {}] Spawned child process {}/{} with OS PID: {}", parent_pid, i + 1, replicas, child_id_u32);

            let serialized_function_info = bincode::serialize(&function_info_clone_for_child)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize FunctionInfo for child {}: {}", i + 1, e)))?;
            
            if let Err(e) = tx_to_child_p.send(serialized_function_info) {
                error!("[PARENT PID {}] Failed to send FunctionInfo to child {} (OS PID {}): {:?}", parent_pid, i + 1, child_id_u32, e);
                child_process.kill().ok();
                child_process.wait().ok();
                return Err(PyValueError::new_err(format!("Failed to send FunctionInfo to child {}: {:?}", i + 1, e)));
            }
            info!("[PARENT PID {}] Sent FunctionInfo to child {} (OS PID {})", parent_pid, i + 1, child_id_u32);

            match rx_from_child_p.recv() {
                Ok(payload) if payload.is_empty() => {
                    info!("[PARENT PID {}] Child process {} (OS PID {}) initialized successfully", parent_pid, i + 1, child_id_u32);
                }
                Ok(_non_empty_payload) => {
                    error!("[PARENT PID {}] Child process {} (OS PID {}) sent unexpected init signal.", parent_pid, i + 1, child_id_u32);
                    child_process.kill().ok();
                    child_process.wait().ok();
                    return Err(PyValueError::new_err(format!(
                        "Child process {} (OS PID {}) sent unexpected init signal", i + 1, child_id_u32
                    )));
                }
                Err(e) => {
                    error!("[PARENT PID {}] Child process {} (OS PID {}) failed to initialize or send signal: {:?}", parent_pid, i + 1, child_id_u32, e);
                    return Err(PyValueError::new_err(format!(
                        "Child process {} (OS PID {}) failed to initialize: {:?}",
                        i + 1, child_id_u32, e
                    )));
                }
            }
            
            let parent_channels = ParentEndChannels { tx_to_child: tx_to_child_p, rx_from_child: rx_from_child_p };
            let replica_info = ReplicaProcessInfo { 
                ipc_channels: parent_channels, 
                child_pid: nix::unistd::Pid::from_raw(child_id_u32 as i32), 
            };
            all_replica_infos.push(Arc::new(Mutex::new(replica_info)));
        }

        info!("[PARENT PID {}] All {} replica processes setup complete", parent_pid, replicas);
        Ok(AsyncPool {
            child_processes: all_replica_infos,
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
                Ok(Ok(pickled_ret)) => { 
                    Python::with_gil(|py_inner| {
                        info!("[PARENT] IPC successful, unpickling result");
                        let cloudpickle = py_inner.import("cloudpickle")?;
                        let loads = cloudpickle.getattr("loads")?;
                        let py_bytes_ret = PyBytes::new(py_inner, &pickled_ret);
                        let bound_obj = loads.call1((py_bytes_ret,))?;
                        Ok(bound_obj.to_object(py_inner)) 
                    })
                }
                Ok(Err(ipc_err_str)) => { 
                    Err(PyValueError::new_err(ipc_err_str))
                }
                Err(join_err) => { 
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
            match replica_arc.try_lock() { 
                Ok(replica_info) => {
                    info!("[PARENT] Terminating child process {} (PID: {})", i, replica_info.child_pid);
                    match nix::sys::signal::kill(replica_info.child_pid, nix::sys::signal::Signal::SIGTERM) {
                        Ok(_) => info!("[PARENT] Sent SIGTERM to child process PID {}", replica_info.child_pid),
                        Err(e) => warn!("[PARENT] Failed to send SIGTERM to child PID {}: {:?}", replica_info.child_pid, e),
                    }
                }
                Err(e) => {
                    warn!("[PARENT] Could not lock replica info for child {} during cleanup: {}. Skipping termination signal for this replica.", i, e);
                }
            }
        }
        self.child_processes.clear();
        info!("[PARENT] Child processes cleanup attempt finished.");
        Ok(())
    }
}

impl Drop for AsyncPool {
    fn drop(&mut self) {
        info!("[PARENT] AsyncPool is being dropped, cleaning up");
        if !self.child_processes.is_empty() { 
            let _ = self.cleanup();
        }
    }
}

#[pymodule]
fn gilboost(m: &Bound<'_, PyModule>) -> PyResult<()> {

    let _ = env_logger::try_init(); 
    info!("[GILBOOST_MODULE_LOAD {}] Initializing gilboost module.", process::id());
    m.add("dummy_attr", "test")?; // Add a dummy attribute
    m.add_class::<AsyncPool>()?;
    m.add_function(wrap_pyfunction!(child, m)?)?;
    
    Ok(())
}
