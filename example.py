import numpy as np
from other_gil import Process

print("Creating process")
@Process.from_signature
def scale(x: np.ndarray, factor: float) -> np.ndarray:
    print(f"Scaling {x} by {factor}")
    return x * factor
print("Process created")
# usage:
arr = np.arange(10, dtype=float)
print("calling scale")
import time
time.sleep(1)
out = scale(arr, 2.5)
print("out)ut:", out)