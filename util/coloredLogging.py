import logging
import sys

COLS = {
    'grey':   '\x1b[90m',
    'red':    '\x1b[31m',
    'green':  '\x1b[32m',
    'blue':   '\x1b[34m',
    'cyan':   '\x1b[36m',
    'purple': '\x1b[35m',
    'yellow': '\x1b[33m',
    'error':  '\x1b[31;1m',
    'reset':  '\x1b[0m'
}


class _ColorFormatter(logging.Formatter):
    errFormat = "%(levelname)s: %(message)s    (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: COLS['grey'] + '%(message)s    (%(filename)s:%(lineno)d)' + COLS['reset'],
        logging.INFO: COLS['blue'] + '%(message)s' + COLS['reset'],
        logging.WARNING: COLS['yellow'] + errFormat + COLS['reset'],
        logging.ERROR: COLS['red'] + errFormat + COLS['reset'],
        logging.CRITICAL: COLS['error'] + errFormat + COLS['reset']
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_colors(logger=logging.getLogger()):
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(_ColorFormatter())

    logger.addHandler(ch)


def printc(col, *args, **kwargs):
    msg = COLS[col] + ' '.join(map(str, args)) + COLS['reset']
    print(msg, **kwargs)


def printHeader(*args, **kwargs):
    print('')
    printc('yellow', '#' * 80)
    print(*args, **kwargs)
    printc('yellow', '#' * 80)
    print('')
