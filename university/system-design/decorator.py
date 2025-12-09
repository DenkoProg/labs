import time

def profiler(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start} seconds")
        return result

    return wrapper


@profiler
def count(n: int):
    for i in range(n):
        time.sleep(1)
        print(i)

count(5)