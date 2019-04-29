# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import json
import logging
import re

from colorama import init as colorama_init
from colorama import Fore, Back, Style


class Log:
    """Class responsible for logging information."""

    # Define header styles
    HEADER_W = [Fore.BLACK, Back.WHITE, Style.BRIGHT]
    HEADER_R = [Fore.WHITE, Back.RED, Style.BRIGHT]
    HEADER_G = [Fore.WHITE, Back.GREEN, Style.BRIGHT]

    @classmethod
    def enable(cls, storage):
        """Initializes the logger.

        Args:
            storage: Storage object.
        """

        # Init colorama to enable colors
        colorama_init()
        # Get deepswarm logger
        cls.logger = logging.getLogger("deepswarm")

        # Create stream handler
        stream_handler = logging.StreamHandler()
        stream_formater = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formater)
        # Add stream handler to logger
        cls.logger.addHandler(stream_handler)

        # Create and setup file handler
        file_handler = logging.FileHandler(storage.current_path / "deepswarm.log")
        file_formater = FileFormatter("%(asctime)s\n%(message)s")
        file_handler.setFormatter(file_formater)
        # Add file handle to logger
        cls.logger.addHandler(file_handler)

        # Set logger level to debug
        cls.logger.setLevel(logging.DEBUG)

    @classmethod
    def header(cls, message, type="WHITE"):
        if type == "RED":
            options = cls.HEADER_R
        elif type == "GREEN":
            options = cls.HEADER_G
        else:
            options = cls.HEADER_W

        cls.info(message.center(80, '-'), options)

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
        # Convert dictionary to nicely formatted JSON
        if isinstance(message, dict):
            message = json.dumps(message, indent=4, sort_keys=True)

        # Convert all objects that are not strings to strings
        if isinstance(message, str) is False:
            message = str(message)

        return ''.join(options) + message + '\033[0m'


class FileFormatter(logging.Formatter):
    """Class responsible for removing ANSI characters from the log file."""

    def plain(self, string):
        # Regex code adapted from Martijn Pieters https://stackoverflow.com/a/14693789
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]|[-]{2,}')
        return ansi_escape.sub('', string)

    def format(self, record):
        message = super(FileFormatter, self).format(record)
        plain_message = self.plain(message)
        separator = '=' * 80
        return ''.join((separator, "\n", plain_message, "\n", separator))
