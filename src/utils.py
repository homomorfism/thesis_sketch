from contextlib import contextmanager
from time import time


@contextmanager
def timer(description: str) -> None:
    start = time()
    yield

    elapsed_time = time() - start
    print(f"{description} executed in: {round(elapsed_time, 2)} sec.")
