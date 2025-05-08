import numpy as np
from other_gil import Process

print("Creating process")
@Process.from_signature
def scale(x: float, factor: float) -> float:
    print(f"Scaling {x} by {factor}")
    return x * factor
print("Process created")
# usage:

print("calling scale")
for i in range(2):
    out = scale(3, 3)
print("output:", out)