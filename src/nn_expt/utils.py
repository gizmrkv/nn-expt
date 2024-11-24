import datetime


def get_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
