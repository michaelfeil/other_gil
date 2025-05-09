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

struct IpcChannelBundle {
    tx: ipc::IpcSender<Vec<u8>>,
    rx: ipc::IpcReceiver<Vec<u8>>,
}

#[pyclass(unsendable)]
struct Process {
    channels: Arc<Mutex<IpcChannelBundle>>,
    function_info: FunctionInfo,
    child_pid: Option<nix::unistd::Pid>,
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
impl Process {
    #[staticmethod]
    #[pyo3(signature=(func))]
    fn from_signature(py: Python<'_>, func: PyObject) -> PyResult<Self> {
        info!("[PARENT] Creating Process from signature");
        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        let module_name = func_ref.getattr("__module__")?.extract::<String>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        info!(
            "[PARENT] Function identified: {}.{}",
            module_name, function_name
        );
        // Pickle the function with cloudpickle.
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

        debug!("[PARENT] Setting up IPC channels");
        let (tx_p, rx_c): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
            ipc::channel().unwrap();
        let (tx_c, rx_p): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) =
            ipc::channel().unwrap();

        // Fork the child process.
        info!("[PARENT] Forking child process");
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                info!("[PARENT] Forked child process with PID: {}", child);
                // Wait for the child's initialization signal.
                match rx_p.recv() {
                    Ok(_) => info!("[PARENT] Child process initialized successfully"),
                    Err(e) => {
                        error!("[PARENT] Child process failed to initialize: {:?}", e);
                        return Err(PyValueError::new_err(format!(
                            "Child process failed to initialize: {:?}",
                            e
                        )))
                    }
                }
                let channels = Arc::new(Mutex::new(IpcChannelBundle { tx: tx_p, rx: rx_p }));
                info!("[PARENT] Process setup complete");
                Ok(Process {
                    channels,
                    function_info,
                    child_pid: Some(child),
                })
            }
            Ok(ForkResult::Child) => {
                info!("[CHILD] Child process forked with PID: {}", process::id());
                // It's good practice to re-initialize the logger in the child process
                // if the parent had initialized it, especially if it involves file handles
                // or other resources that might not be correctly inherited or shared after fork.
                // However, env_logger typically writes to stderr, which should be fine.
                // For more complex logging setups, consider explicit re-initialization.
                unsafe { ffi::PyOS_AfterFork_Child() };
                child_loop(function_info, rx_c, tx_c);
                process::exit(0);
            }
            Err(e) => {
                error!("[PARENT] Fork failed: {}", e);
                return Err(PyValueError::new_err(format!("Fork failed: {}", e)))
            },
        }
    }

    #[pyo3(signature=(*args,))]
    fn __call__<'p>(
        &mut self,
        py: Python<'p>,
        args: Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'p, PyAny>> {
        info!("[PARENT] Calling process function");

        let pickled_args: Vec<u8> = {
            let cloudpickle = py.import("cloudpickle")?;
            let dumps = cloudpickle.getattr("dumps")?;
            let py_bytes = dumps.call1((args,))?;
            py_bytes.extract()?
        };

        let channels_clone = self.channels.clone(); // Clone Arc for the async block

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Perform blocking IPC operations in a separate thread managed by Tokio
            let ipc_task_result = tokio::task::spawn_blocking(move || {
                let guard = channels_clone.lock().map_err(|e| {
                    let msg = format!("Failed to acquire channel lock: {}", e);
                    error!("[PARENT] {}", msg);
                    msg 
                })?;
                trace!("[PARENT] Sending pickled arguments to child");
                guard.tx.send(pickled_args).map_err(|e| {
                    let msg = format!("Failed to send arguments to child: {}", e);
                    error!("[PARENT] {}", msg);
                    msg
                })?;
                debug!("[PARENT] Waiting for pickled return value from child");
                guard.rx.recv().map_err(|e| {
                    let msg = format!("Failed to receive data from child: {}", e);
                    error!("[PARENT] {}", msg);
                    msg
                })
                // This closure returns Result<Vec<u8>, String>
            }).await;

            // ipc_task_result is Result<Result<Vec<u8>, String>, tokio::task::JoinError>
            // The async block must return PyResult<PyObject>
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
        info!("[PARENT] Cleanup called, terminating child process");
        drop(self.channels.clone());

        // Send signal to terminate child if needed
        if let Some(pid) = self.child_pid {
            match nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM) {
                Ok(_) => info!("[PARENT] Sent SIGTERM to child process {}", pid),
                Err(e) => warn!("[PARENT] Failed to send SIGTERM to child {}: {:?}", pid, e),
            }
        }
        Ok(())
    }
}

// Add Drop implementation for Process
impl Drop for Process {
    fn drop(&mut self) {
        info!("[PARENT] Process is being dropped, cleaning up");
        let _ = self.cleanup();
    }
}

#[pymodule]
fn other_gil(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the logger. You can customize this further if needed.
    // For example, to set a default log level if RUST_LOG is not set:
    // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Or simply:
    let _ = env_logger::try_init(); // Use try_init to avoid panic if already initialized
    info!("[MODULE] Initializing other_gil module");
    m.add_class::<Process>()?;
    Ok(())
}
