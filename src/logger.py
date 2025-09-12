import logging
import os


def Logger(name, log_file, logs_dir="logs"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "\n%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        os.makedirs(logs_dir, exist_ok=True)
        logs_file_path = os.path.join(logs_dir, log_file)
        file_handler = logging.FileHandler(logs_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
