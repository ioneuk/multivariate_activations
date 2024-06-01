import logging

import torch.distributed as dist


def create_logger(logging_dir=None):
    """
    Create a logger that writes to stdout and log file.
    """
    if dist.is_initialized() and dist.get_rank() == 0:  # real logger
        handlers = [logging.StreamHandler()]
        if logging_dir:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
