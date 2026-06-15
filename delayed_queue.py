import heapq
import threading
import time
from typing import Any, Callable, Optional


class DelayedTask:
    __slots__ = ("execute_time", "seq", "callback", "args", "kwargs", "cancelled")

    def __init__(
        self,
        execute_time: float,
        seq: int,
        callback: Callable,
        args: tuple,
        kwargs: dict,
    ):
        self.execute_time = execute_time
        self.seq = seq
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.cancelled = False

    def __lt__(self, other: "DelayedTask") -> bool:
        if self.execute_time != other.execute_time:
            return self.execute_time < other.execute_time
        return self.seq < other.seq


class DelayedQueue:
    def __init__(self, worker_count: int = 4):
        self._heap: list[DelayedTask] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True
        self._executor = _ThreadPool(worker_count)
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(
        self, delay: float, callback: Callable, *args: Any, **kwargs: Any
    ) -> DelayedTask:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        execute_time = time.monotonic() + delay
        with self._condition:
            task = DelayedTask(execute_time, self._seq, callback, args, kwargs)
            self._seq += 1
            heapq.heappush(self._heap, task)
            self._condition.notify()
            return task

    def cancel(self, task: DelayedTask) -> bool:
        with self._lock:
            if task.cancelled:
                return False
            task.cancelled = True
            return True

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if wait:
            self._dispatcher.join()
            self._executor.shutdown(wait=True)

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                while self._running:
                    if not self._heap:
                        self._condition.wait()
                        continue
                    now = time.monotonic()
                    task = self._heap[0]
                    if task.cancelled:
                        heapq.heappop(self._heap)
                        continue
                    if task.execute_time <= now:
                        heapq.heappop(self._heap)
                        break
                    timeout = task.execute_time - now
                    self._condition.wait(timeout=timeout)
                else:
                    return

            if not task.cancelled:
                self._executor.submit(task.callback, *task.args, **task.kwargs)


class _ThreadPool:
    def __init__(self, max_workers: int):
        self._max_workers = max_workers
        self._workers: list[threading.Thread] = []
        self._task_queue: list[Optional[tuple]] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("ThreadPool is shut down")
            self._task_queue.append((fn, args, kwargs))
            if len(self._workers) < self._max_workers and (
                len(self._workers) < len(self._task_queue)
            ):
                t = threading.Thread(target=self._worker_loop, daemon=True)
                self._workers.append(t)
                t.start()
            self._condition.notify()

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if wait:
            for t in self._workers:
                t.join()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and not self._task_queue:
                    self._condition.wait()
                if not self._task_queue:
                    if not self._running:
                        return
            with self._condition:
                if not self._task_queue:
                    continue
                fn, args, kwargs = self._task_queue.pop(0)
            try:
                fn(*args, **kwargs)
            except Exception:
                pass


if __name__ == "__main__":
    queue = DelayedQueue(worker_count=4)

    start = time.monotonic()

    def on_fire(name: str, delay: float):
        elapsed = time.monotonic() - start
        print(f"[{elapsed:6.2f}s] task '{name}' fired (expected ~{delay:.1f}s)")

    queue.submit(1.0, on_fire, "A", 1.0)
    queue.submit(3.0, on_fire, "B", 3.0)
    queue.submit(0.5, on_fire, "C", 0.5)
    queue.submit(2.0, on_fire, "D", 2.0)

    cancel_task = queue.submit(2.5, on_fire, "E-cancelled", 2.5)
    queue.cancel(cancel_task)

    print("All tasks submitted. Waiting for execution...")

    time.sleep(5)
    queue.shutdown(wait=True)
    print("DelayedQueue shut down.")
