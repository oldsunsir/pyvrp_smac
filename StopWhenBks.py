import time
from typing import Optional

class StopWhenBks:
    """
    Criterion that stops when cur res == Bks
    """

    def __init__(self, bks : float, maxruntime : int):
        if bks < 0:
            raise ValueError("bks < 0 not understood.")

        self._bks = bks
        self._start_runtime : Optional[float] = None
        self._max_runtime = maxruntime

    def __call__(self, best_cost: float) -> bool:
        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()
        return best_cost <= self._bks or time.perf_counter() - self._start_runtime > self._max_runtime

