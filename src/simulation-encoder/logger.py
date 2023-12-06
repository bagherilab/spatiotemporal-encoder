import logging
import os


class Logger:
    """
    Attributes
    ----------
    exp_name : str
        Name of the experiment.
    log_dir : str
        Directory where the log file is stored.
    format_str : str
        Format string for the logger.
    level : int
        Level of the logger.

    Methods
    -------
    _set_up_logger(format_str: str, level: int)
        Sets up the logger.
    log(msg: str)
        Logs the message.
    """

    def __init__(
        self,
        exp_name: str,
        log_dir: str = "/logs/",
        format_str: str = "%(asctime)s:%(name)s:%(message)s",
        level=logging.INFO,
    ):
        self.log_path = os.path.join(log_dir, f"{exp_name}.log")
        self.exp_name = exp_name

        self.logger = logging.getLogger(self.exp_name)
        self._set_up_logger(format_str, level)

    def _set_up_logger(self, format_str: str, level: int):
        self.logger.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, msg: str):
        self.logger.info(msg)
