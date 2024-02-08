"""Utils for properly logging process termination"""


def write_termination_log(msg: str, file: str="/dev/termination-log") -> None:
    """Writes to the termination logfile."""

    try:
        with open(file, "w") as termination_file:
            termination_file.write(f"{msg}\n")
    except Exception:
        # Ignore any errors writing to the termination logfile.
        # Users can fall back to the stdout logs, and we don't want to pollute
        # those with an error here.
        pass
