import logging
import coloredLogging
from os.path import basename

class CSV2Internal:
    recognized_commands = ['Note_on_c', 'Note_off_c', 'Control_c']
    stepsPerMeasure = 96
    run_count = 0

    def __init__(self, max_channels):
        self.tempo = 0
        self.time_signature = None
        self.csv = []
        self.max_channels = max_channels
        self.clocks_per_quarter_note = 0
        self.note_tracker = [None] * max_channels
        self.drop_count = 0

    def convert(self, csv, filename=None):
        if not filename:
            self.run_count += 1
            filename = f"untitled{self.run_count}.csv"
        logging.debug('Resetting instance variables')
        self.tempo = 0
        self.time_signature = None
        self.csv = [ln.strip() for ln in csv.split('\n')]
        self.drop_count = 0
        self.note_tracker = [None] * self.max_channels

        header = self.csv.pop(0).split(',')
        logging.debug(f'Header: {header}')
        self.clocks_per_quarter_note = int(header[5])
        logging.info('Interleaving channels')
        interleaved = self.interleave()
        logging.info('Parsing interleaved data')
        out = self.parse(interleaved)
        print(f'Done converting: <{filename}>')
        return out

    def interleave(self):
        interleaved = []

        for line in self.csv:
            row = [r.strip() for r in line.split(',')]
            time = int(row[1])
            cmd = row[2]

            if cmd == 'Tempo' and not self.tempo:
                logging.debug('Setting tempo')
                self.tempo = row[3]
            elif cmd == 'Time_signature' and not self.time_signature:
                self.time_signature = [int(row[3]), int(row[4])]
            elif cmd in self.recognized_commands:
                interleaved.append((time, row))

        interleaved.sort()
        return interleaved

    # Converts MIDI clocks to our quantized time model
    def getStepsPerClock(self):
        logging.debug(f'Time signature: {self.time_signature[0]}/{self.time_signature[1]}')
        quarter_notes_per_measure = 4 * self.time_signature[0] / self.time_signature[1]
        clocks_per_measure = self.clocks_per_quarter_note * quarter_notes_per_measure
        clocks_per_step = clocks_per_measure / self.stepsPerMeasure
        logging.debug(f'Clocks per step: {clocks_per_step}')
        return 1 / clocks_per_step

    def updateNote(self, note):
        if note in self.note_tracker:
            index = self.note_tracker.index(note)
            if note.isPress():
                self.note_tracker[index] = note
            else:
                self.note_tracker[index] = None
        elif note.isPress():
            if None in self.note_tracker:
                self.note_tracker[self.note_tracker.index(None)] = note
            else:
                self.drop_count += 1

    def note_array(self):
        out = [[0] * 3] * self.max_channels
        for index, note in enumerate(self.note_tracker):
            if note:
                out[index] = [128, note.pitch, note.velocity]
        return out

    class Note:
        def __init__(self, tuple):
            self.pitch = int(tuple[4])
            self.velocity = int(tuple[5])
            self.channelId = (tuple[0], tuple[3])
            if tuple[2] == 'Note_off_c':
                self.velocity = 0

        def isPress(self):
            return self.velocity

        def isRelease(self):
            return not self.isPress()

        def __eq__(self, other):
            if not isinstance(other, CSV2Internal.Note):
                return False

            return self.pitch == other.pitch and self.channelId == other.channelId

        def __hash__(self):
            return hash((self.pitch, self.channelId))

    def parse(self, interleaved):
        steps_per_clock = self.getStepsPerClock()
        max_clock = interleaved[-1][0]
        max_step = round(max_clock * steps_per_clock)
        logging.debug(f'Max timestep: {max_step}')
        out = [[]] * (max_step + 1)

        sustain_notes = set()
        step_prev = -1
        sustaining = False
        for clock, entry in interleaved:
            step = round(clock * steps_per_clock)
            # Fill in steps we skipped over
            if step > max(0, step_prev):

                out[step_prev:step] = [out[step_prev]] * (step - step_prev)

            # Perform any updates to the notes
            cmd = entry[2]
            if cmd == 'Control_c':
                if int(entry[4]) == 64:
                    if int(entry[5]) > 0:
                        sustaining = True
                    else:
                        sustain_notes.clear()
                        sustaining = False

            if cmd in {'Note_on_c', 'Note_off_c'}:
                note = self.Note(entry)
                if note.isRelease() and sustaining:
                    sustain_notes.add(note)
                else:
                    self.updateNote(note)

            # Add any new step data
            if step != step_prev:
                out[step] = self.note_array()
                step_prev = step

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            out_str = ',\n'.join(str(v) for v in out)
            logging.debug(f'Converted to internal: \n{out_str}\n')
        if self.drop_count > 0:
            logging.warning(f'{self.drop_count} {"Notes" if self.drop_count > 1 else "Note"} dropped')

        return out


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    coloredLogging.apply()
    with open('./test/resources/haydn_7_1.csv') as f:
        converter = CSV2Internal(5)
        converter.convert(f.read(), basename(f.name))


