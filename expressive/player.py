import sounddevice as sd
import logging
from util.coloredLogging import printHeader
from expressive.apu import APU
import util.interruptableSleep as iSleep


def play(exprsco, volume=0.25, device=sd.default.device['output'], tick_rate=24.0):
    try:
        sample_rate = sd.query_devices(device, 'output')['default_samplerate']
        logging.debug(f'Output sample rate: {sample_rate}')
    except Exception as e:
        logging.error('Failed to extract sample rate:')
        raise e

    apu = APU(sample_rate, tick_rate, exprsco)
    track_length = len(exprsco) / tick_rate

    def callback(outdata, frames, time, status):
        if status:
            logging.error(status)
            exit(1)
        apu.callback(outdata, frames)
        outdata[:] *= volume

    try:
        with sd.OutputStream(device=device, channels=1, callback=callback, samplerate=sample_rate):
            printHeader('    Playing audio; press [RETURN] to abort.')
            if iSleep.sleep_or_enter(track_length):
                print('Playback aborted.')
            else:
                print('Done playing.')
            sd.stop()
            return

    except Exception as e:
        logging.error('Failed to play.')
        raise e
