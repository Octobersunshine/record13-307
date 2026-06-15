import heapq
import importlib
import os
import pickle
import sqlite3
import threading
import time
from typing import Any, Callable, Optional


class _TaskStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn_lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delayed_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execute_time REAL NOT NULL,
                callback_key TEXT NOT NULL,
                args BLOB,
                kwargs BLOB,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_task(
        self, execute_time: float, callback_key: str, args: tuple, kwargs: dict
    ) -> int:
        with self._conn_lock:
            cur = self._conn.execute(
                "INSERT INTO delayed_tasks (execute_time, callback_key, args, kwargs, created_at) VALUES (?, ?, ?, ?, ?)",
                (execute_time, callback_key, pickle.dumps(args), pickle.dumps(kwargs), time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def delete_task(self, task_id: int) -> None:
        with self._conn_lock:
            self._conn.execute("DELETE FROM delayed_tasks WHERE id = ?", (task_id,))
            self._conn.commit()

    def load_all_tasks(self) -> list[tuple[int, float, str, tuple, dict]]:
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT id, execute_time, callback_key, args, kwargs FROM delayed_tasks ORDER BY execute_time, id"
            ).fetchall()
        result = []
        for task_id, exec_time, cb_key, args_blob, kwargs_blob in rows:
            try:
                args = pickle.loads(args_blob) if args_blob else ()
                kwargs = pickle.loads(kwargs_blob) if kwargs_blob else {}
            except Exception:
                with self._conn_lock:
                    self._conn.execute("DELETE FROM delayed_tasks WHERE id = ?", (task_id,))
                    self._conn.commit()
                continue
            result.append((task_id, exec_time, cb_key, args, kwargs))
        return result

    def close(self) -> None:
        with self._conn_lock:
            self._conn.close()


class DelayedTask:
    __slots__ = ("execute_time", "seq", "callback", "args", "kwargs", "cancelled", "id")

    def __init__(
        self,
        execute_time: float,
        seq: int,
        callback: Callable,
        args: tuple,
        kwargs: dict,
        task_id: Optional[int] = None,
    ):
        self.execute_time = execute_time
        self.seq = seq
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.cancelled = False
        self.id = task_id

    def __lt__(self, other: "DelayedTask") -> bool:
        if self.execute_time != other.execute_time:
            return self.execute_time < other.execute_time
        return self.seq < other.seq


class DelayedQueue:
    def __init__(self, worker_count: int = 4, db_path: Optional[str] = None):
        self._heap: list[DelayedTask] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True
        self._store: Optional[_TaskStore] = None
        self._callback_registry: dict[str, Callable] = {}
        if db_path is not None:
            self._store = _TaskStore(db_path)
            self._recover_persisted_tasks()
        self._executor = _ThreadPool(worker_count)
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def register_callback(self, key: str, callback: Callable) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("callback key must be a non-empty string")
        with self._lock:
            self._callback_registry[key] = callback

    def submit(
        self,
        delay: float,
        callback: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> DelayedTask:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        execute_time = time.time() + delay
        callback_key = None
        task_id = None
        if self._store is not None:
            callback_key = self._resolve_callback_key(callback)
        with self._condition:
            if self._store is not None:
                task_id = self._store.save_task(execute_time, callback_key, args, kwargs)
            task = DelayedTask(execute_time, self._seq, callback, args, kwargs, task_id=task_id)
            self._seq += 1
            heapq.heappush(self._heap, task)
            self._condition.notify()
            return task

    def cancel(self, task: DelayedTask) -> bool:
        with self._lock:
            if task.cancelled:
                return False
            task.cancelled = True
            if self._store is not None and task.id is not None:
                self._store.delete_task(task.id)
            return True

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if wait:
            self._dispatcher.join()
            self._executor.shutdown(wait=True)
        if self._store is not None:
            self._store.close()

    def _resolve_callback_key(self, callback: Callable) -> str:
        with self._lock:
            for key, cb in self._callback_registry.items():
                if cb is callback:
                    return key
        if hasattr(callback, "__module__") and hasattr(callback, "__qualname__"):
            module = callback.__module__
            qualname = callback.__qualname__
            if module and qualname and "<locals>" not in qualname and "<lambda>" not in qualname:
                return f"{module}:{qualname}"
        raise ValueError(
            "Callback cannot be automatically serialized for persistence. "
            "Register it explicitly via queue.register_callback(key, callback)."
        )

    def _lookup_callback(self, key: str) -> Optional[Callable]:
        with self._lock:
            if key in self._callback_registry:
                return self._callback_registry[key]
        if ":" in key:
            module_name, qualname = key.split(":", 1)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                return None
            obj = module
            for part in qualname.split("."):
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    return None
            return obj
        return None

    def _recover_persisted_tasks(self) -> None:
        rows = self._store.load_all_tasks()
        now = time.time()
        for task_id, exec_time, cb_key, args, kwargs in rows:
            callback = self._lookup_callback(cb_key)
            if callback is None:
                print(f"[WARN] Cannot resolve callback '{cb_key}' for task {task_id}, dropping.")
                self._store.delete_task(task_id)
                continue
            effective_time = max(exec_time, now)
            task = DelayedTask(effective_time, self._seq, callback, args, kwargs, task_id=task_id)
            self._seq += 1
            heapq.heappush(self._heap, task)

    def _dispatch_loop(self) -> None:
        while True:
            task: Optional[DelayedTask] = None
            with self._condition:
                while self._running:
                    if not self._heap:
                        self._condition.wait()
                        continue
                    now = time.time()
                    candidate = self._heap[0]
                    if candidate.cancelled:
                        heapq.heappop(self._heap)
                        continue
                    if candidate.execute_time <= now:
                        task = heapq.heappop(self._heap)
                        break
                    timeout = candidate.execute_time - now
                    self._condition.wait(timeout=timeout)
                else:
                    return

            if self._store is not None and task.id is not None:
                self._store.delete_task(task.id)
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
            except Exception as exc:
                print(f"[WARN] Callback raised exception: {exc}")


if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delayed_queue.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    def on_fire(name: str, delay: float):
        print(f"[{time.strftime('%H:%M:%S')}] task '{name}' fired (expected ~{delay:.1f}s)")

    print("=== Phase 1: submit tasks and shutdown before they fire ===")
    queue = DelayedQueue(worker_count=2, db_path=db_path)
    queue.submit(3.0, on_fire, "A", 3.0)
    queue.submit(5.0, on_fire, "B", 5.0)
    queue.submit(2.0, on_fire, "C", 2.0)
    print("Submitted tasks, shutting down immediately...")
    queue.shutdown(wait=True)

    print("\n=== Phase 2: restart queue (tasks should fire on their schedule) ===")
    queue2 = DelayedQueue(worker_count=2, db_path=db_path)
    queue2.submit(1.5, on_fire, "D-new", 1.5)
    time.sleep(7)
    queue2.shutdown(wait=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Done.")
