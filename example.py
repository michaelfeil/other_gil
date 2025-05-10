from gilboost import AsyncPool
import asyncio
import time


def _sum_pow(base: int, power: int) -> int:
    return sum(
        [1]*base**power
    )

async def sum_pow(base: int, power: int) -> int:
    await asyncio.sleep(0.0001)
    return _sum_pow(base, power)

sum_pow_process = AsyncPool.wraps(sum_pow, replicas=8)
    
# benchmark 2*vanilla_sum_pow vs (vanilla_sum_pow + sum_pow)
async def benchmark(n=8, exp=22):

    start = time.time()

    await asyncio.gather(
        *[sum_pow(2, exp) for _ in range(n)],
    )
    end = time.time()
    
    print(f"Vanilla benchmark finished in {end - start:.2f} seconds")
    start = time.time()
    await asyncio.gather(
        *[sum_pow_process(2, exp) for _ in range(n)],
    )
    end = time.time()
    print(f"Process benchmark finished in {end - start:.2f} seconds")
    

async def main():    
    result = await sum_pow(2, 10)
    print(f"Result: {result}")
    
    print("Starting benchmark")
    await benchmark()
    

if __name__ == "__main__":
    asyncio.run(main())
    print("Main finished")