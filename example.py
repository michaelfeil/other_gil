from other_gil import Process
import asyncio

print("Creating process")
@Process.from_signature
async def sum_pow(base: int, power: int) -> int:
    return sum(
        [1]*base**power
    )
    
async def main():
    print("Starting process")
    result = await sum_pow(2, 10)
    print(f"Result: {result}")
    print("Process finished")

if __name__ == "__main__":
    asyncio.run(main())
    print("Main finished")