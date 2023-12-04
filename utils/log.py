import os
import datetime

class Log:

    def __init__(self, path: str or None):
        self.path = path

    def log(self, message: str, mode: str = "wp", time_format: str or None = "%Y-%m-%d %H:%M:%S"):
        """
        @param mode: "wp" or "w" or "p"
        """

        if self.path is None:
            if mode == "w":
                raise Exception("Missing log path.")
            mode = "p"

        if time_format:
            now = datetime.datetime.now().strftime(time_format)
            message = f"[{now}] {message}"
        else:
            message = f"{message}"

        if "w" in mode:
            with open(self.path, "a") as f:
                f.write(message + '\n')

        if "p" in mode:
            print(message)
