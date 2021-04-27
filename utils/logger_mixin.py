import logging
import sys

import gin


@gin.configurable()
class LoggerMixin:
    """Logger handler, deals with log formatting, naming and streaming.

    Attributes:
        filename: name of the log file.
        log_level: log level of the logger (e.g. DEBUG or INFO).
        shared: if true all the classes using LoggerMixin will share the
            logging format.
        silent: whether to output to the command line.
    """

    shared = False
    silent = False
    log_level = "DEBUG"
    filename = "temp.log"
    redirect_stderr = False

    @property
    def formatter(self):
        """
        Define logger formatter.
        """
        if LoggerMixin.shared:
            format_str = "%(asctime)s | %(message)s"
        else:
            if self.log_level == "INFO":
                format_str = "%(asctime)s | %(name)-10s | %(funcName)s | %(message)s"
            else:
                format_str = "%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s() | %(message)s"
        return logging.Formatter(format_str)

    @property
    def logging(self):
        """
        Initialize the logger and the required handlers.
        """
        if LoggerMixin.shared:
            obj = LoggerMixin
            name = __name__
        else:
            obj = self
            name = self.__class__.__name__

        if not hasattr(obj, "_logger"):
            logging.root.handlers.clear()
            logging.getLogger(__name__).handlers.clear()
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            obj._logger = logger
            if not self.silent:
                self.setup_stream_handler()
                obj._logger.addHandler(LoggerMixin.sh)
            if self.filename:
                self.setup_file_handler(filename=self.filename)
                obj._logger.addHandler(LoggerMixin.fh)

        return obj._logger

    def setup_file_handler(
        self, filename: str = "temp.log", force: bool = False, level: bool = None,
    ):
        """
        Setup file handler.

        Args:
            filename: name of the log file.
            force: whether to force the re-creation of the file handler.
            level: logging level.
        """
        if not hasattr(LoggerMixin, "fh") or force:
            LoggerMixin.fh = logging.FileHandler(
                filename=filename, mode="w", encoding="utf-8",
            )
            level = self.log_level if level is None else level
            LoggerMixin.fh.setLevel(level)
            LoggerMixin.fh.setFormatter(self.formatter)

        if (not hasattr(LoggerMixin, "fh_err") or force) and self.redirect_stderr:
            LoggerMixin.fh_err = logging.FileHandler(
                f"{filename}.err", mode="w", encoding="utf-8"
            )
            level = "WARNING"
            LoggerMixin.fh_err.setLevel(level)
            LoggerMixin.fh_err.setFormatter(self.formatter)

    def setup_stream_handler(self):
        """
        Setup the screen stream handler for the logger.
        """
        if not hasattr(LoggerMixin, "sh"):
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(self.formatter)
            LoggerMixin.sh = sh

        if not hasattr(LoggerMixin, "sh_err") and self.redirect_stderr:
            sh_err = logging.StreamHandler(sys.stderr)
            sh_err.setFormatter(self.formatter)
            LoggerMixin.sh_err = sh_err
