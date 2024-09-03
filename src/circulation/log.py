import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text


def log_table(rich_table):
    """Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    """
    console = Console(width=150)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def setup_logging(level=logging.DEBUG, comm=None):
    handlers = [RichHandler(console=Console(width=200))]
    if comm is not None:
        handlers[0].addFilter(lambda record: 1 if comm.rank == 0 else 0)
    logging.basicConfig(
        level=level,
        handlers=handlers,
    )
