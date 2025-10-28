import logging
import sys


FORMAT = '%(levelname)s: %(asctime)s|[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s'


class Logger(object):

    @staticmethod
    def getLogger(name, logginglevel=logging.DEBUG):
        logger = Logger._loggerSetup(name, logginglevel=logginglevel)
        return logger

    @staticmethod
    def _loggerSetup(filename, logginglevel=logging.INFO):
        logging.basicConfig(
            # format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
            format=FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S %Z',
            stream=sys.stdout
        )

        logger = logging.getLogger(filename)
        logger.setLevel(logginglevel)

        return logger
