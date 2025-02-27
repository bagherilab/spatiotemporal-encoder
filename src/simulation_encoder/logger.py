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
        log_dir: str = "logs/",
        log_name: str = "logs",
        format_str: str = "%(asctime)s:%(name)s:%(message)s",
        level: int = logging.INFO,
        verbose: bool = False,
    ):
        self.experiment_name = ""
        self.model_name = ""
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"{log_name}.log")
        self.logger = logging.getLogger(str(self.experiment_name))
        self.verbose = verbose
        self._set_up_logger(format_str, level)
        self._create_log_file()

    def set_experiment_name(self, experiment_name: str) -> None:
        """
        Sets the name of the experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        """
        self.experiment_name = experiment_name

    def set_model_name(self, model_name: str) -> None:
        """
        Sets the name of the model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        """
        self.model_name = model_name

    def log(self, msg: str) -> None:
        self._log(msg, "info")

    def warning(self, msg: str) -> None:
        self._log(msg, "warning")

    def _log(self, msg: str, level: str) -> None:
        """
        Logs the message at the specified level.

        Parameters
        ----------
        msg : str
            Message to log.
        level : str
            Logging level (e.g., "info", "warning").
        """
        if self.model_name:
            log_msg = f"[{self.experiment_name}][{self.model_name}] {msg}"
        elif self.experiment_name:
            log_msg = f"[{self.experiment_name}] {msg}"
        else:
            log_msg = msg

        if level == "info":
            self.logger.info(log_msg)
        elif level == "warning":
            self.logger.warning(log_msg)

    def _set_up_logger(self, format_str: str, level: int) -> None:
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if self.verbose:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def _create_log_file(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")
