from threading import Thread, Event
from time import sleep

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


class Interrupt_Poller:
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
    

def _tryInterrupt():
    import coloredLogging
    coloredLogging.init_colors()
    interrupter_1 = Interrupt_Poller(4)
    interrupter_2 = Interrupt_Poller(6)
    wasDone1 = False
    wasDone2 = False
    print('Started timers.')
    while interrupter_1.is_alive() or interrupter_2.is_alive():
        sleep(0.1)
        if not (interrupter_1.is_alive() or wasDone1):
            print('1 is done')
            wasDone1 = True
        if not (interrupter_2.is_alive() or wasDone2):
            print('2 is done')
            wasDone2 = True
    print('done')


if __name__ == '__main__':
    _tryInterrupt()
