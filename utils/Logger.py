import logging
import os
import sys

DEFAULT_LOG_FILE = "logs"

class DefaultLogger:
    def __init__(self, path: str = DEFAULT_LOG_FILE, name: str = None, level: int = logging.DEBUG, print_in_terminal: bool = False):
        if name is None:
            raise ValueError("Logger name must be provided")
        
        # Asegura que el directorio exista
        os.makedirs(path, exist_ok=True)
        log_file = os.path.join(path, f"{name}.log")
        
        # Crea y configura el logger interno
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Formatter para los mensajes de log
        formatter = logging.Formatter(
            '[%(asctime)s] - (%(name)s) - |%(levelname)s| : %(message)s'
        )
        
        # Handler de consola, se añade solo si print_in_terminal es True
        if print_in_terminal:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handler para archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Añade el handler de archivo al logger
        self.logger.addHandler(file_handler)
    
    # Métodos de logging
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

# Example usage:
if __name__ == "__main__":
    logger = DefaultLogger(name="test_logger", print_in_terminal=True)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")