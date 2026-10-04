from PySide6.QtCore import QRunnable, QObject, Signal
from traceback import print_exc

class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(object)
    finished = Signal()


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print_exc()
            self.signals.error.emit(e)
        else: 
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
        