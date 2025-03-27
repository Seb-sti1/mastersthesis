"""
This module contains code to help monitor and benchmark the performance of the code.
"""

import threading
import time

import psutil


class RamMonitor:
    def __init__(self):
        self.thread = None
        self.max_ram = 0
        self.sum_ram = 0
        self.init_ram = self.get_current_ram()
        self.count = 0
        self._stop_event = threading.Event()

    def get_current_ram(self) -> float:
        raise NotImplemented

    def monitor_ram(self):
        while not self._stop_event.is_set():
            current_ram = self.get_current_ram() / (1024 * 1024)  # MB
            self.max_ram = max(self.max_ram, current_ram)
            self.sum_ram += current_ram
            self.count += 1
            time.sleep(1)

    def start(self):
        self.thread = threading.Thread(target=self.monitor_ram)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()

    def get_max_ram(self):
        return self.max_ram - self.init_ram

    def get_average_ram(self):
        return self.sum_ram / self.count - self.init_ram


class CPURamMonitor(RamMonitor):
    def __init__(self):
        super().__init__()

    def get_current_ram(self) -> float:
        return psutil.virtual_memory().percent


class GPURamMonitor(RamMonitor):
    def __init__(self, gpu_index=0):
        try:
            import pynvml as cuda
            self.cuda = cuda
            self.cuda.nvmlInit()
        except ImportError:
            raise RuntimeError("pynvml is not installed or GPU is not available.")
        self.gpu_index = gpu_index
        self.handle = self.cuda.nvmlDeviceGetHandleByIndex(self.gpu_index)
        super().__init__()

    def get_current_ram(self) -> float:
        mem_info = self.cuda.nvmlDeviceGetMemoryInfo(self.handle)
        return mem_info.used  # Returns GPU memory usage in bytes

    def __del__(self):
        self.cuda.nvmlShutdown()
