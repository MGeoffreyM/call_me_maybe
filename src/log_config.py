"""Configuration centralisée des logs pour l'application."""

import logging
import sys
import os
from typing import cast, Any
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, Field


TOKEN_LEVEL = 21
logging.addLevelName(TOKEN_LEVEL, "TOKEN")


class AppLogger(logging.Logger):
    """
    Classe de Logger personnalisée.
    Hérite de logging.Logger pour que Mypy reconnaîsse la méthode token().
    """
    def token(self, message: str, *args: Any, **kws: Any) -> None:
        """Méthode personnalisée pour tracer les tokens générés."""
        if self.isEnabledFor(TOKEN_LEVEL):
            self._log(TOKEN_LEVEL, message, args, **kws)


logging.setLoggerClass(AppLogger)


class ColorFormatter(BaseModel):
    """Formateur de log. Injecte des codes ANSI."""
    fmt: str = Field(..., description="Le format de base de la chaîne de log")
    datefmt: str = Field(default="%H:%M:%S", description="Format de l'heure")

    MAGENTA: str = "\x1b[35m"
    BLUE: str = "\x1b[34m"
    CYAN: str = "\x1b[36m"
    YELLOW: str = "\x1b[33m"
    RED: str = "\x1b[31m"
    BOLD_RED: str = "\x1b[31;1m"
    RESET: str = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Applique la coloration ANSI en fonction du niveau de log."""
        level_colors = {
            logging.DEBUG: self.MAGENTA,
            logging.INFO: self.BLUE,
            TOKEN_LEVEL: self.CYAN,      # Association du niveau TOKEN au Cyan
            logging.WARNING: self.YELLOW,
            logging.ERROR: self.RED,
            logging.CRITICAL: self.BOLD_RED
        }
        color = level_colors.get(record.levelno, "")
        colored_format = f"{color}{self.fmt}{self.RESET}"
        native_formatter = logging.Formatter(colored_format,
                                             datefmt=self.datefmt)
        return native_formatter.format(record)


class MainFileFilter(logging.Filter):
    """Filtre pour sys_call_me_maybe.log."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "production"


class ProductionFileFilter(logging.Filter):
    """Filtre pour prod_call_me_maybe.log."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "production":
            return True
        if record.name == "__main__" and record.levelno >= logging.INFO:
            return True
        return False


def setup_logging(console_level: int = logging.INFO) -> None:
    """Initialise la configuration de tous les loggers et fichiers."""
    base_format = "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
    standard_formatter = logging.Formatter(base_format, datefmt="%H:%M:%S")

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    main_file_path = os.path.join(log_dir, "sys_call_me_maybe.log")
    main_file_handler = RotatingFileHandler(
        main_file_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    main_file_handler.setFormatter(standard_formatter)
    main_file_handler.addFilter(MainFileFilter())

    prod_file_path = os.path.join(log_dir, "prod_call_me_maybe.log")
    prod_file_handler = RotatingFileHandler(
        prod_file_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    prod_file_handler.setFormatter(standard_formatter)
    prod_file_handler.addFilter(ProductionFileFilter())

    console_handler = logging.StreamHandler(sys.stdout)
    color_formatter = ColorFormatter(fmt=base_format)
    console_handler.setFormatter(cast(logging.Formatter, color_formatter))

    logging.basicConfig(
        level=console_level,
        handlers=[main_file_handler, prod_file_handler, console_handler],
        force=True
    )

    prod_logger = logging.getLogger("production")
    prod_logger.setLevel(logging.INFO)
    prod_logger.propagate = True
