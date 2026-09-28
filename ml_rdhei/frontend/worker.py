from PySide6.QtCore import QRunnable, QThreadPool, QObject, Slot, Signal

class WorkerSignals(QObject):
    result = Signal(object)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        self.signals.result.emit(result)