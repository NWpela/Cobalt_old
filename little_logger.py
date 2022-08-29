import time

# Class Little_logger created to handle logs in the project

class Little_logger:
    def __init__(self, localisation):
        self.__info_prefix = f"({localisation})[INFO]: "
        self.__warning_prefix = f"({localisation})[WARNING]: "
        self.__error_prefix = f"({localisation})[ERROR]: "
        self.time_start = time.time()

    def set_start(self):
        self.time_start = time.time()

    def info(self, message, with_time=False):
        if with_time:
            message += f" - {round(time.time() - self.time_start, 4)}s -"
        print(self.__info_prefix + message)

    def warning(self, message, with_time=False):
        if with_time:
            message += f" - {round(time.time() - self.time_start, 4)}s -"
        print(self.__warning_prefix + message)

    def error(self, message, with_time=False):
        if with_time:
            message += f" - {round(time.time() - self.time_start, 4)}s -"
        print(self.__error_prefix + message)
