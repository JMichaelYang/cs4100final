import logging

GREY = "\x1b[90;21m"
YELLOW = "\x1b[33;21m"
GREEN = "\x1b[32;21m"
BLUE = "\x1b[34;21m"
PURPLE = "\x1b[35;21m"
CYAN = "\x1b[36;21m"
RED = "\x1b[31;21m"
BOLD_RED = "\x1b[31;1m"
RESET = "\x1b[0m"


class ColorfulFormatter(logging.Formatter):
    format = "%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: GREY + format + RESET,
        logging.INFO: BLUE + format + RESET,
        logging.WARNING: YELLOW + format + RESET,
        logging.ERROR: RED + format + RESET,
        logging.CRITICAL: BOLD_RED + format + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def apply(logger=logging.getLogger()):
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(ColorfulFormatter())

    logger.addHandler(ch)


def printc(col, *args, **kwargs):
    msg = col + ' '.join(map(str, args)) + RESET
    print(msg, **kwargs)
