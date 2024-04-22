import os
import logging


class ExperimentLogger:
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
        uuid: str = "none",
        log_dir: str = "logs/",
        format_str: str = "%(asctime)s:%(name)s:%(message)s",
        level: int = logging.INFO,
        verbose: bool = False,
    ):
        self.uuid = uuid
        self.log_path = os.path.join(log_dir, f"{self.uuid}.log")
        self.logger = logging.getLogger(str(self.uuid))
        self.verbose = verbose
        self._create_log_file()
        self._set_up_logger(format_str, level)

    def _set_up_logger(self, format_str: str, level: int) -> None:
        self.logger.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _create_log_file(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, msg: str) -> None:
        """
        Logs the message.

        Parameters
        ----------
        msg : str
            Message to log.

        """
        self.logger.info(msg)
        if self.verbose:
            print(msg)

    def warning(self, msg: str) -> None:
        """
        Logs a warning message.

        Parameters
        ----------
        msg : str
            Warning message to log.

        """
        self.logger.warning(msg)
        if self.verbose:
            print(msg)
