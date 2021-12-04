import sys
import logging
import util.coloredLogging as cl


def player():
    from expressive.codec import loadFile
    from expressive.player import play
    
    cl.printHeader('Testing expressive.player')
    
    data = loadFile('./test/resources/295_SilverSurfer_02_03SectionStart.exprsco.pkl')
    play(data)


def interruptableSleep():
    import util.interruptableSleep as iSleep
    from time import sleep
    
    cl.printHeader('Testing util.interruptableSleep')
    
    interrupter_1 = iSleep.InterruptPoller(2)
    interrupter_2 = iSleep.InterruptPoller(3)
    
    was_done1 = False
    was_done2 = False
    print('Started timers.')
    while interrupter_1.is_alive() or interrupter_2.is_alive():
        sleep(0.1)
        if not (interrupter_1.is_alive() or was_done1):
            print('1 is done')
            was_done1 = True
        if not (interrupter_2.is_alive() or was_done2):
            print('2 is done')
            was_done2 = True
    print('done')


tests = {
    '--player': player,
    '--interrupt': interruptableSleep
}


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    cl.init_colors()
    
    for arg in sys.argv[1:]:
        tests[arg]()


if __name__ == '__main__':
    main()
