use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyFunction, PyModule, PyTuple};
use pyo3::Python;
use ipc_channel::ipc;
use std::sync::{Arc, Mutex};
use nix::unistd::{fork, ForkResult};
use pyo3::ffi;
use std::process;

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
    println!("[CHILD] Starting child loop with PID: {}", process::id());
    // Signal parent that we have initialized.
    if let Err(e) = tx.send(vec![]) {
        println!("[CHILD] Failed to send initialization signal: {:?}", e);
    }

    Python::with_gil(|py| {
        println!("[CHILD] Unpickling function with cloudpickle");
        let cloudpickle = match py.import("cloudpickle") {
            Ok(m) => m,
            Err(e) => {
                println!("[CHILD] Error importing cloudpickle: {:?}", e);
                return;
            }
        };
        let dumps = cloudpickle.getattr("dumps").unwrap();
        let loads = cloudpickle.getattr("loads").unwrap();

        let func = match || -> PyResult<Bound<'_, PyFunction>> {
            let unpickled = loads.call1((PyBytes::new(py, &function_info.pickled_func),))?;
            let func = unpickled.downcast::<PyFunction>()?;
            Ok(func.clone())
        }() {
            Ok(f) => f,
            Err(e) => {
                println!("[CHILD] Failed to unpickle function: {:?}", e);
                return;
            }
        };
        println!("[CHILD] Function unpickled successfully: {}", function_info.function_name);
        

        loop {
            println!("[CHILD] Waiting for pickled arguments from parent...");
            let pickled_args = match rx.recv() {
                Ok(bytes) => bytes,
                Err(e) => {
                    println!("[CHILD] Error receiving arguments: {:?}, exiting", e);
                    break;
                }
            };
            // Unpickle the args to a Python object (expecting a tuple)
            let args_obj = match loads.call1((PyBytes::new(py, &pickled_args),)) {
                Ok(o) => o,
                Err(e) => {
                    println!("[CHILD] Error unpickling args: {:?}", e);
                    continue;
                }
            };
            let args_tuple = match args_obj.downcast::<PyTuple>() {
                Ok(t) => t,
                Err(e) => {
                    println!("[CHILD] Expected tuple of arguments, got error: {:?}", e);
                    continue;
                }
            };
            println!("[CHILD] Calling function with unpickled arguments");
            let ret = match func.call(args_tuple, None) {
                Ok(r) => r,
                Err(e) => {
                    println!("[CHILD] Error calling function: {:?}", e);
                    continue;
                }
            };
            // Pickle the return value and send it over IPC.
            
            let pickled_ret: Vec<u8> = match dumps.call1((ret,)) {
                Ok(py_bytes) => match py_bytes.extract() {
                    Ok(b) => b,
                    Err(e) => {
                        println!("[CHILD] Error extracting pickled return: {:?}", e);
                        continue;
                    }
                },
                Err(e) => {
                    println!("[CHILD] Error pickling return value: {:?}", e);
                    continue;
                }
            };
            if let Err(e) = tx.send(pickled_ret) {
                println!("[CHILD] Failed to send return value: {:?}", e);
            }
        } // loop
        println!("[CHILD] Exiting child loop");
    });
}

#[pymethods]
impl Process {
    #[staticmethod]
    #[pyo3(signature=(func))]
    fn from_signature(py: Python<'_>, func: PyObject) -> PyResult<Self> {
        println!("[PARENT] Creating Process from signature");
        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        let module_name = func_ref.getattr("__module__")?.extract::<String>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        println!("[PARENT] Function identified: {}.{}", module_name, function_name);
        // Pickle the function with cloudpickle.
        let cloudpickle = py.import("cloudpickle")?;
        let pickled_func: Vec<u8> = cloudpickle.getattr("dumps")?
            .call1((func.clone_ref(py),))?
            .extract()?;
        let function_info = FunctionInfo {
            module_name,
            function_name,
            pickled_func,
        };

        println!("[PARENT] Setting up IPC channels");
        let (tx_p, rx_c): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) = ipc::channel().unwrap();
        let (tx_c, rx_p): (ipc::IpcSender<Vec<u8>>, ipc::IpcReceiver<Vec<u8>>) = ipc::channel().unwrap();

        // Fork the child process.
        println!("[PARENT] Forking child process");
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                println!("[PARENT] Forked child process with PID: {}", child);
                // Wait for the child's initialization signal.
                match rx_p.recv() {
                    Ok(_) => println!("[PARENT] Child process initialized successfully"),
                    Err(e) => return Err(PyValueError::new_err(format!("Child process failed to initialize: {:?}", e))),
                }
                let channels = Arc::new(Mutex::new(IpcChannelBundle { tx: tx_p, rx: rx_p }));
                println!("[PARENT] Process setup complete");
                Ok(Process { 
                    channels, 
                    function_info,
                    child_pid: Some(child),
                })
            },
            Ok(ForkResult::Child) => {
                println!("[CHILD] Child process forked with PID: {}", process::id());
                unsafe { ffi::PyOS_AfterFork_Child() };
                child_loop(function_info, rx_c, tx_c);
                process::exit(0);
            },
            Err(e) => return Err(PyValueError::new_err(format!("Fork failed: {}", e))),
        }
    }

    #[pyo3(signature=(*args,))]
    fn __call__(&mut self, py: Python<'_>, args: Bound<'_, PyTuple>) -> PyResult<PyObject> {
        println!("[PARENT] Calling process function");
        // Pickle the arguments using cloudpickle.
        let pickled_args: Vec<u8> = Python::with_gil(|py| {
            let cloudpickle = py.import("cloudpickle")?;
            let dumps = cloudpickle.getattr("dumps")?;
            let py_bytes = dumps.call1((args,))?;
            py_bytes.extract()
        })?;
        let pickled_ret: Vec<u8> = py.allow_threads(|| {
            let lock = self.channels.lock().map_err(|e| e.to_string())?;
            lock.tx.send(pickled_args).map_err(|e| e.to_string())?;
            // Wait for the child to send back the pickled return value.
            lock.rx.recv().map_err(|e| e.to_string())
        })
        .map_err(|err_str: String| PyValueError::new_err(err_str))?;
        // Unpickle the return value.
        let py_ret = Python::with_gil(|py| {
            let cloudpickle = py.import("cloudpickle")?;
            let loads = cloudpickle.getattr("loads")?;
            let obj = loads.call1((PyBytes::new(py, &pickled_ret),))?;
            let val = obj.extract()
                .map_err(|e| PyValueError::new_err(format!("Failed to unpickle return value: {:?}", e)));
            val
        })?;
        println!("[PARENT] Call complete, returning result");
        Ok(py_ret)
    }
    
    fn cleanup(&mut self) -> PyResult<()> {
        println!("[PARENT] Cleanup called, terminating child process");
        drop(self.channels.clone());

        // Send signal to terminate child if needed
        if let Some(pid) = self.child_pid {
            match nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM) {
                Ok(_) => println!("[PARENT] Sent SIGTERM to child process"),
                Err(e) => println!("[PARENT] Failed to send SIGTERM: {:?}", e),
            }
        }
        Ok(())
    }
}

// Add Drop implementation for Process
impl Drop for Process {
    fn drop(&mut self) {
        println!("[PARENT] Process is being dropped, cleaning up");
        let _ = self.cleanup();
    }
}

#[pymodule]
fn other_gil(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("[MODULE] Initializing other_gil module");
    m.add_class::<Process>()?;
    Ok(())
}