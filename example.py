import numpy as np
from other_gil import Process

print("Creating process")
@Process.from_signature
def scale(x: np.ndarray, factor: float) -> np.ndarray:
    print(f"Scaling {x} by {factor}")
    return x * factor
print("Process created")
# usage:
arr = np.arange(10)
print("calling scale")
for i in range(2):
    out = scale(arr, 3)
print("output:", out)