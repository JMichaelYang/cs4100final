import numpy as np


def _note2freq(note, sample_rate):
    return np.power(2, (note - 69) / 12) * 440 / sample_rate


class _PulseGen:
    dutyCycles = [.125, .25, .5, .75]
    
    def __init__(self):
        self.enable = False
        self.duty = 0
        self.t = 0
        self.f = 0
        self.amp = 0
    
    def sample(self):
        if not self.enable:
            return 0
        self.t += self.f
        p = self.t - int(self.t)
        p2 = self.t + self.duty
        p2 -= int(p2)
        return (p2 - p) * self.amp
    
    def setParams(self, sample_rate, note, amp, timbre):
        self.enable = note and amp
        self.f = _note2freq(note, sample_rate)
        self.duty = self.dutyCycles[timbre]
        self.amp = amp / 16


class _TriGen:
    def __init__(self):
        self.enable = False
        self.t = 0
        self.f = 0
    
    def sample(self):
        if not self.enable:
            return 0
        self.t += self.f
        p = self.t - int(self.t)
        return 4 * abs(p-0.5)-1
    
    def setParams(self, sample_rate, note, amp, timbre):
        if note:
            self.f = _note2freq(note, sample_rate)
            self.enable = True
        else:
            self.f = 0
            self.enable = False
            self.t = 0.25


class _NoiseGen:
    freqLookup = [
        0x004, 0x008, 0x010, 0x020,
        0x040, 0x060, 0x080, 0x0A0,
        0x0CA, 0x0FE, 0x17C, 0x1FC,
        0x2FA, 0x3F8, 0x7F2, 0xFE4
    ]
    
    def __init__(self):
        self.feedback = 1
        self.mode = 0
        self.t = 0
        self.reset = 4
        self.f = 0
        self.amp = 0
        self.enable = False
        self.out = 0
    
    def clockRegister(self):
        self.out = (self.feedback ^ (self.feedback >> (5 if self.mode else 1))) & 1
        self.feedback = (self.feedback >> 1) & ~(1 << 14)
        self.feedback |= (self.out << 14)
    
    def sample(self):
        if not self.enable:
            return 0
        self.t -= 1
        if self.t < 0:
            self.t += self.reset + 1
            self.clockRegister()
        return (0.5 - self.out) * self.amp
    
    def setParams(self, sample_rate, note, amp, timbre):
        self.amp = amp / 16
        self.mode = timbre
        if note:
            self.reset = self.freqLookup[note - 1] * (sample_rate / 41943040)
            self.enable = True
        else:
            self.enable = False
        

class APU:
    def __init__(self, sample_rate, tick_rate, data):
        self.sample_rate = sample_rate
        self.frame_reset = sample_rate / tick_rate
        self.frame_clock_counter = self.frame_reset
        
        self.generators = [_PulseGen(), _PulseGen(), _TriGen(), _NoiseGen()]
        self.data = data
        self.frame = 0
        self.done = False
        
    def callback(self, outdata, frames):
        for time in range(frames):
            self.frame_clock_counter -= 1
            if self.frame_clock_counter < 0 and not self.done:  # Update frame
                self.frame_clock_counter += self.frame_reset
                self.frame += 1
                
                if self.frame < len(self.data):
                    step = self.data[self.frame]
                    for i, gen in enumerate(self.generators):
                        gen.setParams(self.sample_rate, *step[i])
                else:
                    self.done = True
  
            if not self.done:
                outdata[time] = sum([gen.sample() for gen in self.generators])
            else:
                outdata[time] = 0
        