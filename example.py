import numpy as np
from other_gil import Process

print("Creating process")
@Process.from_signature
def sum_pow(base: int, power: int) -> int:
    return sum(
        [1]*base**power
    )
    
print("Process created")
# usage:

print("calling sum_pow")
for i in range(2):
    out = sum_pow(2, 22)
print("output:", out)