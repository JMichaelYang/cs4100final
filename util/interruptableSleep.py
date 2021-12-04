from threading import Thread, Event

_startEvent = Event()
_inputEvent = Event()


def _pollInput():
    global _startEvent
    global _inputEvent
    
    while True:
        _startEvent.wait()
        input()
        print("\033[A\033[A")
        _inputEvent.set()
        
        _inputEvent.clear()
        _startEvent.clear()


worker = Thread(target=_pollInput)
worker.setDaemon(True)
worker.start()


class InterruptPoller:
    def __init__(self, timeout):
        """
        A polled object that will remain 'alive' until [ENTER] is pressed, or for a maximum of [timeout] seconds.
        :param timeout: the maximum lifespan of the object.
        """
        self.status = 0
        self.worker = self.worker = Thread(target=self.wait, args=[timeout])
        self.worker.setDaemon(True)
        self.worker.start()
    
    def wait(self, timeout):
        global _startEvent
        global _inputEvent
        
        _startEvent.set()
        _inputEvent.wait(timeout)
        self.status = 1 + int(_inputEvent.is_set())
    
    def getStatus(self):
        """
        Gets the current status of the interrupt object
        :return: 0 if alive, 1 if timed out, and 2 if interrupted.
        """
        return self.status
    
    def is_alive(self):
        return self.status == 0
    

def sleep_or_enter(timeout):
    """
    Sleeps until [ENTER] is pressed, or for a maximum of [timeout] seconds.
    :param timeout: The maximum seconds to sleep for
    :return: true if interrupted
    """
    global _startEvent
    global _inputEvent
    
    _startEvent.set()
    return _inputEvent.wait(timeout)
