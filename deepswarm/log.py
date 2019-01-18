# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License


import json
import logging
from colorama import init as colorama_init
from colorama import Fore, Back, Style


class Log:
    # Define header styles
    HEADER_W = [Fore.BLACK, Back.WHITE, Style.BRIGHT]
    HEADER_R = [Fore.WHITE, Back.RED, Style.BRIGHT]
    HEADER_G = [Fore.WHITE, Back.GREEN, Style.BRIGHT]

    @classmethod
    def enable(cls):
        # Init colorama to enable colors
        colorama_init()
        # Create logger
        cls.logger = logging.getLogger("deepswarm")
        logger_handler = logging.StreamHandler()
        logger_handler.setFormatter(logging.Formatter('%(message)s'))
        cls.logger.addHandler(logger_handler)
        cls.logger.setLevel(logging.DEBUG)

    @classmethod
    def header(cls, message, type="WHITE"):
        if type == "RED":
            options = cls.HEADER_R
        elif type == "GREEN":
            options = cls.HEADER_G
        else:
            options = cls.HEADER_W

        # formated_message = cls.create_message(message.center(80, '-'), options)
        cls.info(message.center(80, '-'), options)
        # cls.logger.info(formated_message)

    @classmethod
    def debug(cls, message, options=[Fore.CYAN]):
        formated_message = cls.create_message(message, options)
        cls.logger.debug(formated_message)

    @classmethod
    def info(cls, message, options=[Fore.GREEN]):
        formated_message = cls.create_message(message, options)
        cls.logger.info(formated_message)

    @classmethod
    def warning(cls, message, options=[Fore.YELLOW]):
        formated_message = cls.create_message(message, options)
        cls.logger.warning(formated_message)

    @classmethod
    def error(cls, message, options=[Fore.MAGENTA]):
        formated_message = cls.create_message(message, options)
        cls.logger.error(formated_message)

    @classmethod
    def critical(cls, message, options=[Fore.RED, Style.BRIGHT]):
        formated_message = cls.create_message(message, options)
        cls.logger.critical(formated_message)

    @classmethod
    def create_message(cls, message, options):
        # Convert dictionary to nicely formated JSON
        if isinstance(message, dict):
            message = json.dumps(message, indent=4, sort_keys=True)
        # Convert all objects that are not strings to string
        if isinstance(message, str) is False:
            message = str(message)
        return ''.join(options) + message + '\033[0m'
