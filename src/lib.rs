use ipc_channel::ipc::{self, IpcOneShotServer, IpcSender};
use log::{debug, error, info, trace, warn};
use nix::unistd::Pid;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFunction, PyTuple};
use pyo3::Python;
use serde::{Deserialize, Serialize};
use std::env;
use std::process::{self, Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Clone, Serialize, Deserialize)]
struct FunctionInfo {
    function_name: String,
    pickled_func: Vec<u8>,
}

// Parent's view of IPC channels for one replica
struct ParentEndChannels {
    tx_to_child: ipc::IpcSender<Vec<u8>>, // Parent sends arguments to child
    rx_from_child: ipc::IpcReceiver<Vec<u8>>, // Parent receives results from child
}

// Information about a single replica process
struct ReplicaProcessInfo {
    ipc_channels: ParentEndChannels,
    child_pid: Pid,
}
// Helper for parent to create setup servers for one child
struct IpcSetupEndpoints {
    p2c_rendezvous_server: IpcOneShotServer<IpcSender<Vec<u8>>>,
    p2c_rendezvous_server_name: String,
    c2p_data_server: IpcOneShotServer<Vec<u8>>,
    c2p_data_server_name: String,
}

fn create_ipc_setup_endpoints() -> PyResult<IpcSetupEndpoints> {
    let (p2c_server, p2c_name) = IpcOneShotServer::<IpcSender<Vec<u8>>>::new().map_err(|e| {
        PyValueError::new_err(format!(
            "Parent: Failed to create P->C rendezvous server: {}",
            e
        ))
    })?;
    let (c2p_server, c2p_name) = IpcOneShotServer::<Vec<u8>>::new().map_err(|e| {
        PyValueError::new_err(format!("Parent: Failed to create C->P data server: {}", e))
    })?;
    Ok(IpcSetupEndpoints {
        p2c_rendezvous_server: p2c_server,
        p2c_rendezvous_server_name: p2c_name,
        c2p_data_server: c2p_server,
        c2p_data_server_name: c2p_name,
    })
}

// Helper to attempt to send a pickled Python error back to the parent.
fn try_send_pickled_py_error_to_parent(
    tx_to_parent: &ipc::IpcSender<Vec<u8>>,
    dumps: &Bound<'_, PyAny>, // cloudpickle.dumps
    error_to_pickle: PyObject, // The Python error object to pickle
    child_pid: u32,
    context_msg: &str, // Short description of when the error occurred
) {
    match dumps.call1((error_to_pickle,)) {
        Ok(pickled_error_bytes_obj) => match pickled_error_bytes_obj.extract::<Vec<u8>>() {
            Ok(pickled_error_bytes) => {
                if let Err(e) = tx_to_parent.send(pickled_error_bytes) {
                    error!(
                        "[CHILD PID {}] While {}: Failed to send pickled Python error to parent: {:?}",
                        child_pid, context_msg, e
                    );
                }
            }
            Err(e) => {
                error!(
                    "[CHILD PID {}] While {}: Failed to extract bytes from pickled Python error: {:?}",
                    child_pid, context_msg, e
                );
            }
        },
        Err(e) => {
            error!(
                "[CHILD PID {}] While {}: Failed to pickle Python error for parent: {:?}",
                child_pid, context_msg, e
            );
        }
    }
}

// Helper to receive and deserialize FunctionInfo.
// Returns Err(()) if the child should exit due to an error in this phase.
fn receive_and_deserialize_function_info(
    rx_from_parent: &ipc::IpcReceiver<Vec<u8>>,
    tx_to_parent: &ipc::IpcSender<Vec<u8>>,
    child_pid: u32,
) -> FunctionInfo {
    let function_info: FunctionInfo = match rx_from_parent.recv() {
        Ok(bytes) => match bincode::deserialize(&bytes) {
            Ok(fi) => fi,
            Err(e) => {
                error!(
                    "[CHILD PID {}] Failed to deserialize FunctionInfo: {:?}. Exiting.",
                    child_pid, e
                );
                let err_msg = format!("Child: Deserialize FunctionInfo error: {}", e);
                let serialized_err =
                    bincode::serialize(&Result::<(), String>::Err(err_msg)).unwrap_or_default();
                if tx_to_parent.send(serialized_err).is_err() {
                    error!(
                        "[CHILD PID {}] Also failed to send deserialization error to parent.",
                        child_pid
                    );
                }
                process::exit(1);
            }
        },
        Err(e) => {
            error!(
                "[CHILD PID {}] Failed to receive FunctionInfo: {:?}. Exiting.",
                child_pid, e
            );
            process::exit(1);
        }
    };
    function_info
}

// New synchronous function to handle the processing loop
fn python_processing_loop_synchronous(
    rx_from_parent: ipc::IpcReceiver<Vec<u8>>, // Takes ownership
    tx_to_parent: ipc::IpcSender<Vec<u8>>,   // Takes ownership
    function_info: FunctionInfo,
    child_pid: u32,
) -> Result<(), String> { // Represents loop termination: Ok for graceful (if ever), Err for fatal.
    // One-time Python objects setup (done once, holding GIL)
    let (py_function_obj, is_coroutine, py_dumps_fn, py_loads_fn, py_event_loop_obj_option): (
        PyObject,      // func
        bool,          // is_coroutine
        PyObject,      // cloudpickle.dumps
        PyObject,      // cloudpickle.loads
        Option<PyObject>, // asyncio event loop
    ) = Python::with_gil(|py| {
        let setup_res: PyResult<_> = (|| {
            info!(
                "[CHILD PID {}] Python GIL acquired for initial setup. Unpickling function '{}'.",
                child_pid, function_info.function_name
            );
            let cloudpickle = py.import("cloudpickle")?;
            let asyncio = py.import("asyncio")?;
            let inspect = py.import("inspect")?;
            let dumps_fn = cloudpickle.getattr("dumps")?.to_object(py);
            let loads_fn = cloudpickle.getattr("loads")?.to_object(py);

            let func_bytes = PyBytes::new(py, &function_info.pickled_func);
            let func_obj = loads_fn.bind(py).call1((func_bytes,))?.to_object(py);

            let is_coro = inspect
                .getattr("iscoroutinefunction")?
                .call1((func_obj.bind(py),))?
                .extract::<bool>()?;

            let event_loop_obj = if is_coro {
                Some(asyncio.call_method0("new_event_loop")?.to_object(py))
            } else {
                None
            };
            info!(
                "[CHILD PID {}] Function '{}' unpickled. is_coroutine: {}. Initial setup complete.",
                child_pid, function_info.function_name, is_coro
            );
            Ok((func_obj, is_coro, dumps_fn, loads_fn, event_loop_obj))
        })();
        setup_res
    }).map_err(|e| format!("[CHILD PID {}] Python setup failed: {:?}", child_pid, e))?;

    // Main processing loop
    loop {
        debug!("[CHILD PID {}] Waiting for pickled arguments from parent (GIL NOT HELD)...", child_pid);
        // 1. Receive data (blocking IPC, GIL NOT HELD)
        let pickled_args = match rx_from_parent.recv() {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("[CHILD PID {}] Error receiving arguments from parent: {:?}. Terminating loop.", child_pid, e);
                return Err(format!("IPC receive error: {}", e));
            }
        };

        // 2. Acquire GIL for Python operations
        let gil_result: PyResult<Option<Vec<u8>>> = Python::with_gil(|py| {
            let func = py_function_obj.bind(py);
            let dumps = py_dumps_fn.bind(py);
            let loads = py_loads_fn.bind(py);

            let args_tuple_bound = match loads.call1((PyBytes::new(py, &pickled_args),)) {
                Ok(obj) => match obj.downcast_into::<PyTuple>() {
                    Ok(tuple) => tuple,
                    Err(e) => {
                        error!("[CHILD PID {}] Failed to downcast unpickled args to PyTuple: {:?}. Skipping request.", child_pid, e);
                        try_send_pickled_py_error_to_parent( &tx_to_parent, dumps, PyErr::from(e).into_py(py), child_pid, "downcasting args");
                        return Ok(None); // Signal to continue outer Rust loop
                    }
                },
                Err(e) => {
                    error!("[CHILD PID {}] Error unpickling arguments: {:?}. Skipping request.", child_pid, e);
                    try_send_pickled_py_error_to_parent( &tx_to_parent, dumps, e.into_py(py), child_pid, "unpickling args");
                    return Ok(None); // Signal to continue outer Rust loop
                }
            };

            trace!("[CHILD PID {}] Calling function '{}' with unpickled arguments", child_pid, function_info.function_name);
            let call_result_obj = match func.call(args_tuple_bound.as_borrowed(), None) {
                Ok(r) => r,
                Err(e) => {
                    error!("[CHILD PID {}] Error calling user function '{}': {:?}. Sending error to parent.", child_pid, function_info.function_name, e);
                    try_send_pickled_py_error_to_parent(&tx_to_parent, dumps, e.into_py(py), child_pid, "calling user function");
                    return Ok(None);
                }
            };

            let final_ret_obj = if is_coroutine {
                debug!("[CHILD PID {}] Function '{}' is a coroutine, awaiting result.", child_pid, function_info.function_name);
                let event_loop = py_event_loop_obj_option.as_ref().unwrap().bind(py);
                match event_loop.call_method1("run_until_complete", (call_result_obj,)) {
                    Ok(awaited_result) => awaited_result,
                    Err(e) => {
                        error!("[CHILD PID {}] Error awaiting coroutine for function '{}': {:?}. Sending error to parent.", child_pid, function_info.function_name, e);
                        try_send_pickled_py_error_to_parent(&tx_to_parent, dumps, e.into_py(py), child_pid, "awaiting coroutine");
                        return Ok(None);
                    }
                }
            } else {
                call_result_obj
            };

            match dumps.call1((final_ret_obj.as_ref(),)) {
                Ok(py_bytes_obj) => match py_bytes_obj.extract::<Vec<u8>>() {
                    Ok(b) => Ok(Some(b)),
                    Err(e) => {
                        error!("[CHILD PID {}] Error extracting bytes from pickled return value for function '{}': {:?}. Sending error to parent.", child_pid, function_info.function_name, e);
                        try_send_pickled_py_error_to_parent(&tx_to_parent, dumps, e.into_py(py), child_pid, "extracting pickled return");
                        Ok(None)
                    }
                },
                Err(e) => {
                    error!("[CHILD PID {}] Error pickling return value for function '{}': {:?}. Sending error to parent.", child_pid, function_info.function_name, e);
                    try_send_pickled_py_error_to_parent(&tx_to_parent, dumps, e.into_py(py), child_pid, "pickling return value");
                    Ok(None)
                }
            }
        }); // GIL released

        match gil_result {
            Ok(Some(pickled_ret_bytes)) => {
                // 3. Send data (blocking IPC, GIL NOT HELD)
                if let Err(e) = tx_to_parent.send(pickled_ret_bytes) {
                    error!("[CHILD PID {}] Failed to send result to parent: {:?}. Terminating loop.", child_pid, e);
                    return Err(format!("IPC send error: {}", e));
                }
                trace!("[CHILD PID {}] Successfully sent result for function '{}' to parent.", child_pid, function_info.function_name);
            }
            Ok(None) => {
                // Per-request error was handled by sending a pickled Python error. Continue loop.
                debug!("[CHILD PID {}] Handled per-request error, continuing loop.", child_pid);
                continue;
            }
            Err(py_err) => {
                // Fatal PyErr from the Python::with_gil block.
                error!("[CHILD PID {}] Fatal Python error during processing: {:?}. Terminating loop.", child_pid, py_err);
                return Err(format!("Fatal Python error: {:?}", py_err));
            }
        }
    }
    // Note: Loop is infinite. Ok(()) would be for a designed graceful exit.
}

async fn child_loop(
    rx_from_parent: ipc::IpcReceiver<Vec<u8>>, // Takes ownership
    tx_to_parent: ipc::IpcSender<Vec<u8>>,   // Takes ownership
) {
    let child_pid = process::id();
    info!("[CHILD PID {}] Starting child loop", child_pid);

    // receive_and_deserialize_function_info uses references, but it exits on error.
    // It's called once.
    let function_info =
        receive_and_deserialize_function_info(&rx_from_parent, &tx_to_parent, child_pid);
    // If the above call fails, the process will exit due to its internal logic.

    // Spawn the synchronous processing loop in a blocking-safe thread.
    // The `rx_from_parent`, `tx_to_parent`, and `function_info` are moved into the closure.
    let loop_join_result = tokio::task::spawn_blocking(move || {
        python_processing_loop_synchronous(
            rx_from_parent,
            tx_to_parent,
            function_info, // FunctionInfo is Clone, but here it's moved.
            child_pid,
        )
    })
    .await;

    match loop_join_result {
        Ok(Ok(())) => {
            // This case implies python_processing_loop_synchronous returned Ok(()),
            // which means a graceful loop termination (not currently implemented in the infinite loop).
            info!(
                "[CHILD PID {}] Python processing loop completed gracefully. Exiting.",
                child_pid
            );
        }
        Ok(Err(e_str)) => {
            // python_processing_loop_synchronous returned an error string (IPC or fatal Python error).
            error!(
                "[CHILD PID {}] Python processing loop terminated with error: {}. Exiting.",
                child_pid, e_str
            );
        }
        Err(join_err) => {
            // The spawn_blocking task itself panicked or was cancelled.
            error!(
                "[CHILD PID {}] Python processing loop task failed: {:?}. Exiting.",
                child_pid, join_err
            );
        }
    }

    info!("[CHILD PID {}] Exiting child loop and process.", child_pid);
    process::exit(0); // Ensure child process exits.
}

async fn child_process_entry_point(
    p2c_rendezvous_server_name: String,
    c2p_data_server_name: String,
) -> PyResult<()> {
    info!(
        "[CHILD_SPAWN_ENTRY PID {}] Connecting to parent IPC servers",
        std::process::id()
    );

    let (sender_for_parent_to_use, rx_from_parent) = ipc::channel::<Vec<u8>>().map_err(|e| {
        PyValueError::new_err(format!("Child: Failed to create P->C channel: {}", e))
    })?;

    let conn_to_p2c_rendezvous =
        IpcSender::connect(p2c_rendezvous_server_name.clone()).map_err(|e| {
            PyValueError::new_err(format!(
                "Child: Failed to connect to P->C rendezvous server ({}): {}",
                p2c_rendezvous_server_name, e
            ))
        })?;

    conn_to_p2c_rendezvous
        .send(sender_for_parent_to_use)
        .map_err(|e| {
            PyValueError::new_err(format!(
                "Child: Failed to send P->C sender to parent: {}",
                e
            ))
        })?;
    info!(
        "[CHILD_SPAWN_ENTRY PID {}] P->C channel setup complete. rx_from_parent is ready.",
        std::process::id()
    );

    let tx_to_parent =
        IpcSender::<Vec<u8>>::connect(c2p_data_server_name.clone()).map_err(|e| {
            PyValueError::new_err(format!(
                "Child: Failed to connect C->P sender to data server ({}): {}",
                c2p_data_server_name, e
            ))
        })?;
    info!(
        "[CHILD_SPAWN_ENTRY PID {}] C->P channel setup complete. tx_to_parent is ready.",
        std::process::id()
    );

    if let Err(e) = tx_to_parent.send(vec![]) {
        let err_msg = format!("Child: Failed to send initialization signal: {:?}", e);
        error!("[CHILD_SPAWN_ENTRY PID {}] {}", std::process::id(), err_msg);
        return Err(PyValueError::new_err(err_msg));
    }
    info!(
        "[CHILD_SPAWN_ENTRY PID {}] Sent initialization signal to parent.",
        std::process::id()
    );

    pyo3::prepare_freethreaded_python();

    info!(
        "[CHILD_SPAWN_ENTRY PID {}] IPC fully established. Calling child_loop.",
        std::process::id()
    );

    child_loop(rx_from_parent, tx_to_parent).await;

    Ok(())
}

#[pyfunction]
pub fn _child(
    py: Python,
    p2c_rendezvous_server_name: String,
    c2p_data_server_name: String,
) -> PyResult<Bound<PyAny>> {
    info!(
        "[GILBOOST_CHILD] P2C Rendezvous Server: {}",
        p2c_rendezvous_server_name
    );
    info!("[GILBOOST_CHILD] C2P Data Server: {}", c2p_data_server_name);

    pyo3_async_runtimes::tokio::future_into_py(py, async {
        let _ = child_process_entry_point(p2c_rendezvous_server_name, c2p_data_server_name).await;
        Ok(())
    })
}

#[pyclass()]
struct AsyncPool {
    child_processes: Vec<Arc<Mutex<ReplicaProcessInfo>>>,
    next_replica_idx: Arc<Mutex<usize>>,
}

#[pymethods]
impl AsyncPool {
    #[staticmethod]
    #[pyo3(signature=(func, *, replicas = 8, python_executable = None))]
    fn wraps(
        py: Python<'_>,
        func: PyObject,
        replicas: usize,
        python_executable: Option<String>,
    ) -> PyResult<Self> {
        if replicas == 0 {
            return Err(PyValueError::new_err(
                "Number of replicas must be at least 1",
            ));
        }
        let parent_pid = process::id();
        info!(
            "[PARENT PID {}] Creating AsyncPool with {} replicas.",
            parent_pid, replicas
        );

        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        let cloudpickle = py.import("cloudpickle")?;
        let pickled_func_bytes: Vec<u8> = cloudpickle
            .getattr("dumps")?
            .call1((func.clone_ref(py),))?
            .extract()?;

        let function_info = FunctionInfo {
            function_name,
            pickled_func: pickled_func_bytes,
        };

        let mut child_processes_info = Vec::with_capacity(replicas);
        // this line selects the wrong interpreter if venv is active. How to improve TODO
        let resolved_python_executable = match python_executable {
            Some(path) => std::path::PathBuf::from(path),
            None => {
                let sys = py.import("sys")?;
                let executable_path_str: String = sys.getattr("executable")?.extract()?;
                std::path::PathBuf::from(executable_path_str)
            }
        };
        // make sure path exists
        if !resolved_python_executable.exists() && !resolved_python_executable.is_file() {
            return Err(PyValueError::new_err(format!(
                "Python executable path does not exist: {:?}",
                resolved_python_executable
            )));
        };

        info!(
            "[PARENT PID {}] Current executable for spawning children: {:?}",
            parent_pid, resolved_python_executable
        );

        for i in 0..replicas {
            let replica_id = i + 1;
            debug!(
                "[PARENT PID {}] Setting up IPC for replica {}",
                parent_pid, replica_id
            );

            let setup_endpoints = create_ipc_setup_endpoints()?;

            let function_info_clone = function_info.clone();

            info!(
                "[PARENT PID {}] Spawning childs process {}/{}",
                parent_pid, replica_id, replicas
            );
            let mut cmd = Command::new(&resolved_python_executable);
            cmd.arg("-c")
                .arg(format!(
                    r#"
import os, sys, traceback, asyncio
try:
    import gilboost 
    async def forever():
        await gilboost._child("{}", "{}")
    asyncio.run(forever())
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"#,
                    setup_endpoints.p2c_rendezvous_server_name,
                    setup_endpoints.c2p_data_server_name
                ))
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());

            if let Ok(rust_log_val) = env::var("RUST_LOG") {
                cmd.env("RUST_LOG", rust_log_val);
            }
            if let Ok(python_path_val) = env::var("PYTHONPATH") {
                cmd.env("PYTHONPATH", python_path_val);
            }

            let child_handle = cmd.spawn().map_err(|e| {
                PyValueError::new_err(format!(
                    "Parent: Failed to spawn child process {}/{}: {}",
                    replica_id, replicas, e
                ))
            })?;

            let child_actual_pid = child_handle.id();
            info!(
                "[PARENT PID {}] Spawned child {} (OS PID {}) with P2C Server: {}, C2P Server: {}",
                parent_pid,
                replica_id,
                child_actual_pid,
                setup_endpoints.p2c_rendezvous_server_name,
                setup_endpoints.c2p_data_server_name
            );

            let (_rx_on_p2c_rendezvous, tx_to_child) = setup_endpoints
                .p2c_rendezvous_server
                .accept()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Parent: Failed to accept on P->C rendezvous for child {}: {:?}",
                        replica_id, e
                    ))
                })?;
            info!(
                "[PARENT PID {}] Accepted P->C rendezvous from child {}",
                parent_pid, replica_id
            );

            let (rx_from_child, initial_signal_from_child) =
                setup_endpoints.c2p_data_server.accept().map_err(|e| {
                    PyValueError::new_err(format!(
                        "Parent: Failed to accept on C->P data server for child {}: {:?}",
                        replica_id, e
                    ))
                })?;
            info!(
                "[PARENT PID {}] Accepted C->P data connection from child {}",
                parent_pid, replica_id
            );

            if initial_signal_from_child.is_empty() {
                info!(
                    "[PARENT PID {}] Received correct initialization signal from child {}",
                    parent_pid, replica_id
                );
            } else {
                error!(
                    "[PARENT PID {}] Received unexpected initialization signal from child {}: {:?}",
                    parent_pid, replica_id, initial_signal_from_child
                );
                return Err(PyValueError::new_err(format!(
                    "Parent: Bad init signal from child {}",
                    replica_id
                )));
            }

            let serialized_function_info =
                bincode::serialize(&function_info_clone).map_err(|e| {
                    PyValueError::new_err(format!(
                        "Parent: Failed to serialize FunctionInfo: {}",
                        e
                    ))
                })?;

            tx_to_child.send(serialized_function_info).map_err(|e| {
                PyValueError::new_err(format!(
                    "Parent: Failed to send FunctionInfo to child {}: {:?}",
                    replica_id, e
                ))
            })?;
            info!(
                "[PARENT PID {}] Sent FunctionInfo to child {}",
                parent_pid, replica_id
            );

            child_processes_info.push(Arc::new(Mutex::new(ReplicaProcessInfo {
                ipc_channels: ParentEndChannels {
                    tx_to_child,
                    rx_from_child,
                },
                child_pid: Pid::from_raw(child_actual_pid as i32),
            })));
        }

        info!(
            "[PARENT PID {}] All {} child processes initialized.",
            parent_pid, replicas
        );
        Ok(AsyncPool {
            child_processes: child_processes_info,
            next_replica_idx: Arc::new(Mutex::new(0)),
        })
    }

    #[pyo3(signature=(*args,))]
    fn __call__<'p>(
        &mut self,
        py: Python<'p>,
        args: Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let args_pyobject = args.to_object(py);

        // Clone Arcs for moving into the async block
        let next_replica_idx_clone = self.next_replica_idx.clone();
        let child_processes_clone = self.child_processes.clone(); // Clones the Vec of Arcs

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            
            let ipc_task_result = tokio::task::spawn_blocking(move || {
                // --- Start of moved logic ---
                let mut idx_guard = next_replica_idx_clone.lock().unwrap();
                let current_idx = *idx_guard;
                *idx_guard = (current_idx + 1) % child_processes_clone.len();
                drop(idx_guard);

                let replica_info_arc = child_processes_clone[current_idx].clone();

                info!(
                    "[PARENT] Dispatching call to replica index: {}",
                    current_idx
                );

                let pickled_args: Vec<u8> = Python::with_gil(|py_blocking| {
                    let cloudpickle = py_blocking.import("cloudpickle")?;
                    let dumps = cloudpickle.getattr("dumps")?;
                    // Bind the PyObject to the current GIL context for pickling
                    let py_bytes = dumps.call1((args_pyobject.bind(py_blocking),))?;
                    py_bytes.extract()
                })?; // Propagates PyErr if pickling fails
                // --- End of moved logic ---

                let replica_info_guard = replica_info_arc.lock().unwrap();

                info!(
                    "[PARENT] Worker {}: Acquired lock, dispatching to child PID: {}",
                    current_idx, replica_info_guard.child_pid
                );

                replica_info_guard
                    .ipc_channels
                    .tx_to_child
                    .send(pickled_args)
                    .map_err(|e| {
                        PyValueError::new_err(format!("Failed to send args to child: {:?}", e))
                    })?;

                let pickled_result_bytes = replica_info_guard
                    .ipc_channels
                    .rx_from_child
                    .recv()
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to receive result from child: {:?}",
                            e
                        ))
                    })?;
                Ok(pickled_result_bytes)
            })
            .await;

            match ipc_task_result {
                Ok(Ok(pickled_ret)) => Python::with_gil(|py_inner| {
                    info!("[PARENT] IPC successful, unpickling result");
                    let cloudpickle = py_inner.import("cloudpickle")?;
                    let loads = cloudpickle.getattr("loads")?;
                    let py_bytes_ret = PyBytes::new(py_inner, &pickled_ret);
                    let bound_obj: Bound<'_, PyAny> = loads.call1((py_bytes_ret,))?;
                    Ok(bound_obj.to_object(py_inner))
                }),
                Ok(Err(ipc_py_err)) => {
                    // ipc_py_err is already a PyErr
                    Err(ipc_py_err) // Propagate the existing PyErr
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
        info!(
            "[PARENT] Cleanup called, terminating {} child process(es)",
            self.child_processes.len()
        );

        for (i, replica_arc) in self.child_processes.iter().enumerate() {
            match replica_arc.try_lock() {
                Ok(replica_info) => {
                    info!(
                        "[PARENT] Terminating child process {} (PID: {})",
                        i, replica_info.child_pid
                    );
                    match nix::sys::signal::kill(
                        replica_info.child_pid,
                        nix::sys::signal::Signal::SIGTERM,
                    ) {
                        Ok(_) => info!(
                            "[PARENT] Sent SIGTERM to child process PID {}",
                            replica_info.child_pid
                        ),
                        Err(e) => warn!(
                            "[PARENT] Failed to send SIGTERM to child PID {}: {:?}",
                            replica_info.child_pid, e
                        ),
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
    debug!(
        "[GILBOOST_MODULE_LOAD {}] Initializing gilboost module.",
        process::id()
    );
    m.add_class::<AsyncPool>()?;
    m.add_function(wrap_pyfunction!(_child, m)?)?;
    Ok(())
}
