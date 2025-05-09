from other_gil import Process
import asyncio
import time

print("Creating process")

def _sum_pow(base: int, power: int) -> int:
    return sum(
        [1]*base**power
    )

@Process.from_signature
async def sum_pow(base: int, power: int) -> int:
    await asyncio.sleep(0.1)
    return _sum_pow(base, power)
    
async def vanilla_sum_pow(base: int, power: int) -> int:
    await asyncio.sleep(0.1)
    return _sum_pow(base, power)

# benchmark 2*vanilla_sum_pow vs (vanilla_sum_pow + sum_pow)
async def benchmark(n=2, exp=28):

    start = time.time()

    await asyncio.gather(
        *[vanilla_sum_pow(2, exp) for _ in range(2*n)],
    )
    end = time.time()
    
    print(f"Vanilla benchmark finished in {end - start:.2f} seconds")
    start = time.time()
    await asyncio.gather(
        *[sum_pow(2, exp) for _ in range(n)],
        *[vanilla_sum_pow(2, exp) for _ in range(n)],
    )
    end = time.time()
    print(f"Process benchmark finished in {end - start:.2f} seconds")
    

async def main():    
    print("Starting process")
    result = await sum_pow(2, 10)
    print(f"Result: {result}")
    print("Process finished")
    
    print("Starting benchmark")
    await benchmark()
    

if __name__ == "__main__":
    asyncio.run(main())
    print("Main finished")