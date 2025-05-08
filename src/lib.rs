use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyTuple, PyFunction, PyModule, PyBytes};
use pyo3::{PyAny, PyObject, Python, IntoPy};
use pythonize::{pythonize, depythonize};
use numpy::IntoPyArray;
use numpy::PyArrayDyn;
use numpy::PyArrayMethods;
use serde_json::Value as JsonValue;
use tempfile::tempfile;
use bincode::config;
use memmap2::{MmapMut, MmapOptions};
use ipc_channel::ipc;
use std::sync::Arc;
// New imports for fork
use nix::unistd::{fork, ForkResult};
use pyo3::ffi;
use std::process;

// -------- 1) Type descriptors --------
#[derive(Clone)]
enum ArgSpec {
    NdArray { dtype: String, len: Option<usize> },
    Primitive(String),
    Serde,
}

#[derive(Clone)]
enum RetSpec {
    NdArray { dtype: String, len: Option<usize> },
    Primitive(String),
    Serde,
}

// -------- 2) Shared-memory helper --------
fn create_shm(size: usize) -> MmapMut {
    let file = tempfile().unwrap();
    file.set_len(size as u64).unwrap();
    unsafe { MmapOptions::new().map_mut(&file).unwrap() }
}

// -------- 3) Parsing annotations --------
// Update parse_arg_spec to handle Python type objects
fn parse_arg_spec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<ArgSpec> {
    // Get string representation of the type object
    let type_str = obj.repr()?.extract::<String>()?;
    
    if type_str.contains("ndarray") || type_str.contains("array") {
        Ok(ArgSpec::NdArray { dtype: "f64".to_string(), len: None })
    } else if type_str.contains("int") {
        Ok(ArgSpec::Primitive("int".to_string()))
    } else if type_str.contains("float") {
        Ok(ArgSpec::Primitive("float".to_string()))
    } else {
        Ok(ArgSpec::Serde)
    }
}

// Update parse_ret_spec to pass through the Python context
fn parse_ret_spec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<RetSpec> {
    match parse_arg_spec(py, obj)? {
        ArgSpec::NdArray { dtype, len } => Ok(RetSpec::NdArray { dtype, len }),
        ArgSpec::Primitive(k) => Ok(RetSpec::Primitive(k)),
        ArgSpec::Serde => Ok(RetSpec::Serde),
    }
}

// -------- 4) Marshalling arguments --------
fn marshal_args(
    py: Python<'_>,
    specs: &[ArgSpec],
    maps: &[Arc<MmapMut>],
    tuple: &Bound<'_, PyTuple>, 
) -> PyResult<()> {
    let args = tuple.iter().collect::<Vec<_>>();
    for ((spec, map), arg) in specs.iter().zip(maps.iter()).zip(args.iter()) {
        match spec {
            ArgSpec::NdArray { .. } => {
                // First convert to float64 array explicitly
                let numpy = py.import("numpy")?;
                let arr_obj = numpy.getattr("asarray")?.call1((arg, "float64"))?;
                
                // Now downcast to PyArray
                let arr = arr_obj.downcast::<PyArrayDyn<f64>>()?;
                let slice = unsafe { arr.as_slice()? };
                
                // Copy to shared memory
                let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut f64, slice.len()) };
                dest.copy_from_slice(slice);
            }
            ArgSpec::Primitive(kind) if kind == "int" => {
                let v: i64 = arg.extract()?;
                let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, 8) };
                dest.copy_from_slice(&v.to_le_bytes());
            }
            ArgSpec::Primitive(kind) if kind == "float" => {
                let v: f64 = arg.extract()?;
                let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, 8) };
                dest.copy_from_slice(&v.to_le_bytes());
            }
            ArgSpec::Serde => {
                let val: JsonValue = depythonize(arg)?;
                let bin = bincode::serde::encode_to_vec(&val, config::legacy())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let len = bin.len().min(map.len());
                let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, len) };
                dest.copy_from_slice(&bin[..len]);
            }
            _ => return Err(PyValueError::new_err("Unsupported ArgSpec")),
        }
    }
    Ok(())
}

// -------- 5) Unmarshalling args in child --------
fn unmarshal_args(
    py: Python<'_>,
    specs: &[ArgSpec],
    maps: &[Arc<MmapMut>],
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::with_capacity(specs.len());
    for (spec, map) in specs.iter().zip(maps.iter()) {
        let obj = match spec {
            ArgSpec::NdArray { .. } => {
                let n = map.len() / std::mem::size_of::<f64>();
                let slice = unsafe { std::slice::from_raw_parts(map.as_ptr() as *const f64, n) };
                slice.to_vec().into_pyarray(py).to_object(py)
            }
            ArgSpec::Primitive(kind) if kind == "int" => {
                let mut buf = [0u8; 8]; buf.copy_from_slice(&map[..8]);
                let v: i64 = i64::from_le_bytes(buf);
                v.into_py(py)
            }
            ArgSpec::Primitive(kind) if kind == "float" => {
                let mut buf = [0u8; 8]; buf.copy_from_slice(&map[..8]);
                let v = f64::from_le_bytes(buf);
                v.into_py(py)
            }
            ArgSpec::Serde => {
                let len = map.iter().position(|&b| b == 0).unwrap_or(map.len());
                let slice = &map[..len];
                let val: JsonValue = bincode::serde::decode_from_slice(slice, config::legacy())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .0;
                pythonize(py, &val)?.into_py(py)
            }
            _ => return Err(PyValueError::new_err("Unsupported ArgSpec")),
        };
        out.push(obj);
    }
    Ok(out)
}

// -------- 6) Marshalling return in child --------
fn marshal_ret(
    spec: &RetSpec,
    map: &Arc<MmapMut>,
    ret: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match spec {
        RetSpec::NdArray { .. } => {
            let arr = ret.downcast::<PyArrayDyn<f64>>()?;
            let slice = unsafe { arr.as_slice()? };
            let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut f64, slice.len()) };
            dest.copy_from_slice(slice);
        }
        RetSpec::Primitive(kind) if kind == "int" => {
            let v: i64 = ret.extract()?;
            let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, 8) };
            dest.copy_from_slice(&v.to_le_bytes());
        }
        RetSpec::Primitive(kind) if kind == "float" => {
            let v: f64 = ret.extract()?;
            let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, 8) };
            dest.copy_from_slice(&v.to_le_bytes());
        }
        RetSpec::Serde => {
            let val: JsonValue = depythonize(ret)?;
            let bin = bincode::serde::encode_to_vec(&val, config::legacy())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let len = bin.len().min(map.len());
            let dest = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, len) };
            dest.copy_from_slice(&bin[..len]);
        }
        _ => return Err(PyValueError::new_err("Unsupported RetSpec")),
    }
    Ok(())
}

// -------- 7) Child event loop with improved logging and signal handling --------
#[derive(Clone)]
struct FunctionInfo {
    module_name: String,
    function_name: String,
    pickled_func: Vec<u8>,  // Add this field to store the pickled function
}

fn child_loop(
    function_info: FunctionInfo,
    specs: Vec<ArgSpec>,
    ret_spec: RetSpec,
    in_maps: Vec<Arc<MmapMut>>,
    out_map: Arc<MmapMut>,
    rx: ipc::IpcReceiver<()>,
    tx: ipc::IpcSender<()>,
) {
    println!("[CHILD] Starting child loop with PID: {}", process::id());
    
    match tx.send(()) {
        Ok(_) => println!("[CHILD] Sent initialization signal to parent"),
        Err(e) => println!("[CHILD] Failed to signal parent: {:?}", e),
    }
    
    // In forked process, use regular Python::with_gil instead of embedded interpreter
    Python::with_gil(|py| {
        println!("[CHILD] Acquired GIL for child process");
        
        // Unpickle the function using cloudpickle
        println!("[CHILD] Unpickling function with cloudpickle");
        let func = match || -> PyResult<Bound<'_, PyFunction>> {
            let cloudpickle = PyModule::import(py, "cloudpickle")?;
            let unpickled = cloudpickle.getattr("loads")?.call1((PyBytes::new(py, &function_info.pickled_func),))?;
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
            // Check for Python interrupts
            if py.check_signals().is_err() {
                println!("[CHILD] Received Python interrupt signal, exiting loop");
                // Try to unblock parent if it's waiting
                let _ = tx.send(());
                break;
            }
            
            println!("[CHILD] Waiting for message from parent");
            match rx.recv() {
                Ok(_) => {
                    println!("[CHILD] Message received, processing");
                    
                    // Unmarshal args with error logging
                    let args = match unmarshal_args(py, &specs, &in_maps) {
                        Ok(a) => a,
                        Err(e) => {
                            println!("[CHILD] Error unmarshaling args: {:?}", e);
                            // Notify parent of failure
                            let _ = tx.send(());
                            continue;
                        }
                    };
                    
                    println!("[CHILD] Args unmarshaled, calling function");
                    
                    // Create a proper Python tuple from the args vector
                    let args_tuple = match PyTuple::new(py, &args) {
                        Ok(tuple) => tuple,
                        Err(e) => {
                            println!("[CHILD] Error creating args tuple: {:?}", e);
                            // Notify parent of failure
                            let _ = tx.send(());
                            continue;
                        }
                    };

                    // Call the function with the tuple properly unpacked as args
                    let ret_py = match func.call(args_tuple, None) {
                        Ok(r) => r,
                        Err(e) => {
                            println!("[CHILD] Error calling function: {:?}", e);
                            // Notify parent of failure
                            let _ = tx.send(());
                            continue;
                        }
                    };
                    
                    println!("[CHILD] Function executed, marshaling return value");
                    
                    // Marshal return with error logging
                    match marshal_ret(&ret_spec, &out_map, &ret_py) {
                        Ok(_) => println!("[CHILD] Return value marshaled successfully"),
                        Err(e) => println!("[CHILD] Error marshaling return: {:?}", e),
                    }
                    
                    println!("[CHILD] Sending completion signal to parent");
                    
                    // Send with error handling
                    match tx.send(()) {
                        Ok(_) => println!("[CHILD] Signal sent to parent"),
                        Err(e) => println!("[CHILD] Failed to signal parent: {:?}", e),
                    }
                }
                Err(e) => {
                    println!("[CHILD] Error receiving message: {:?}, exiting loop", e);
                    break;
                }
            }
            
            // Periodically check for interrupts
            if py.check_signals().is_err() {
                println!("[CHILD] Interrupt detected, exiting loop");
                break;
            }
        }
        println!("[CHILD] Exiting child loop");
    });
}

// -------- 8) Process class with improved logging and error handling --------
#[pyclass(unsendable)]
struct Process {
    specs: Vec<ArgSpec>,
    ret_spec: RetSpec,
    in_maps: Vec<Arc<MmapMut>>,
    out_map: Arc<MmapMut>,
    tx: ipc::IpcSender<()>,
    rx: ipc::IpcReceiver<()>,
    function_info: FunctionInfo,
    child_pid: Option<nix::unistd::Pid>,  // Track child process PID
}

#[pymethods]
impl Process {
    #[staticmethod]
    #[pyo3(signature = (func))]
    fn from_signature(py: Python<'_>, func: PyObject) -> PyResult<Self> {
        println!("[PARENT] Creating Process from signature");
        let func_ref = func.bind(py).downcast::<PyFunction>()?;
        
        // Extract module and function names
        let module_name = func_ref.getattr("__module__")?.extract::<String>()?;
        let function_name = func_ref.getattr("__name__")?.extract::<String>()?;
        
        println!("[PARENT] Function identified: {}.{}", module_name, function_name);
        
        // Pickle the function using cloudpickle
        println!("[PARENT] Pickling function with cloudpickle");
        let cloudpickle = py.import("cloudpickle")?;
        let pickled_func = cloudpickle.getattr("dumps")?.call1((func.clone_ref(py),))?.extract::<Vec<u8>>()?;
        
        // Create FunctionInfo with pickled function
        let function_info = FunctionInfo {
            module_name,
            function_name,
            pickled_func,
        };
        
        let binding = func_ref.getattr("__annotations__")?;
        let ann = binding.downcast::<PyDict>()?;
        
        let mut specs = Vec::new();
        let mut ret_spec = RetSpec::Serde;
        
        println!("[PARENT] Parsing function annotations");
        for (k, v) in ann.iter() {
            let name: &str = k.extract()?;
            if name == "return" {
                ret_spec = parse_ret_spec(py, &v)?;
                println!("[PARENT] Found return annotation");
            } else {
                specs.push(parse_arg_spec(py, &v)?);
                println!("[PARENT] Found arg annotation for: {}", name);
            }
        }
        
        println!("[PARENT] Creating shared memory for {} arguments", specs.len());
        let in_maps: Vec<Arc<MmapMut>> = specs.iter()
            .map(|s| Arc::new(create_shm(match s { ArgSpec::NdArray {..} => 8*1024, _ => 4*1024 })))
            .collect();
            
        let out_map = Arc::new(create_shm(match ret_spec { RetSpec::NdArray {..} => 8*1024, _ => 4*1024 }));
        println!("[PARENT] Setting up IPC channels");
        let (tx_p, rx_c) = ipc::channel().unwrap();
        let (tx_c, rx_p) = ipc::channel().unwrap();
        
        let specs_clone = specs.clone();
        let ret_spec_clone = ret_spec.clone();
        let in_maps_clone = in_maps.clone();
        let out_map_clone = out_map.clone();
        let function_info_clone = function_info.clone();
        
        // REPLACE thread spawn with fork
        println!("[PARENT] Forking child process");
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                println!("[PARENT] Forked child process with PID: {}", child);
                
                // Wait for child initialization
                println!("[PARENT] Waiting for child process to initialize");
                match rx_p.try_recv_timeout(std::time::Duration::from_secs(20)) {
                    Ok(_) => println!("[PARENT] Child process initialized successfully"),
                    Err(e) => return Err(PyValueError::new_err(format!("Child process failed to initialize: {:?}", e))),
                }
                
                println!("[PARENT] Process setup complete");
                Ok(Process { 
                    specs, ret_spec, in_maps, out_map, 
                    tx: tx_p, rx: rx_p, function_info,
                    child_pid: Some(child) 
                })
            },
            Ok(ForkResult::Child) => {
                // In child process
                println!("[CHILD] Child process forked with PID: {}", process::id());
                
                // Fix Python's internal state after fork
                unsafe { ffi::PyOS_AfterFork_Child() };
                
                // Run child loop directly
                child_loop(
                    function_info_clone, specs_clone, ret_spec_clone, 
                    in_maps_clone, out_map_clone, rx_c, tx_c
                );
                
                // Exit the child process
                process::exit(0);
            },
            Err(e) => return Err(PyValueError::new_err(format!("Fork failed: {}", e))),
        }
    }

    #[pyo3(signature = (*args,))]
    fn __call__(&mut self, py: Python<'_>, args: Bound<'_, PyTuple>) -> PyResult<PyObject> {
        println!("[PARENT] Calling process function");
        
        match marshal_args(py, &self.specs, &self.in_maps, &args) {
            Ok(_) => println!("[PARENT] Arguments marshaled successfully"),
            Err(e) => return Err(PyValueError::new_err(format!("Failed to marshal arguments: {:?}", e))),
        }
        // todo allow threads while waiting for the child to process to avoid blocking
        
        println!("[PARENT] Sending message to child process");
        match self.tx.send(()) {
            Ok(_) => println!("[PARENT] Message sent successfully"),
            Err(e) => return Err(PyValueError::new_err(format!("Failed to send to child process: {:?}", e))),
        }
        
        println!("[PARENT] Waiting for response from child");
        match self.rx.recv() {
            Ok(_) => println!("[PARENT] Response received from child"),
            Err(e) => return Err(PyValueError::new_err(format!("Failed to receive from child process: {:?}", e))),
        }
        
        println!("[PARENT] Processing return value");
        let result = match &self.ret_spec {
            RetSpec::NdArray {..} => {
                println!("[PARENT] Handling ndarray return");
                let n = self.out_map.len()/std::mem::size_of::<f64>();
                let slice = unsafe { std::slice::from_raw_parts(self.out_map.as_ptr() as *const f64, n) };
                slice.to_vec().into_pyarray(py).into_py(py)
            }
            RetSpec::Primitive(kind) if kind=="int" => {
                println!("[PARENT] Handling int return");
                let mut buf=[0u8;8]; buf.copy_from_slice(&self.out_map[..8]);
                let v=i64::from_le_bytes(buf);
                v.to_object(py)
            }
            RetSpec::Primitive(kind) if kind=="float" => {
                println!("[PARENT] Handling float return");
                let mut buf=[0u8;8]; buf.copy_from_slice(&self.out_map[..8]);
                let v=f64::from_le_bytes(buf);
                v.to_object(py)
            }
            RetSpec::Serde => {
                println!("[PARENT] Handling serde return");
                let len = self.out_map.iter().position(|&b|b==0).unwrap_or(self.out_map.len());
                let slice=&self.out_map[..len];
                let val: JsonValue = match bincode::serde::decode_from_slice(slice, config::legacy()) {
                    Ok((v, _)) => v,
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to decode return value: {:?}", e))),
                };
                match pythonize(py, &val) {
                    Ok(obj) => obj.to_object(py),
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to pythonize return value: {:?}", e))),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported RetSpec")),
        };
        
        println!("[PARENT] Call complete, returning result");
        Ok(result)
    }
    
    fn cleanup(&mut self) -> PyResult<()> {
        println!("[PARENT] Cleanup called, terminating child process");
        drop(self.tx.clone());
        
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