from datetime import datetime
import subprocess


def timeslice_cast(timerange):
    """Ensure the timerange is a slice of datetime objects"""
    timerange_start = timerange.start
    timerange_stop = timerange.stop
    if isinstance(timerange_start, str):
        timerange_start = datetime.fromisoformat(timerange_start)
    if isinstance(timerange_stop, str):
        timerange_stop = datetime.fromisoformat(timerange_stop)

    return slice(timerange_start, timerange_stop)

# %% git revision
def get_git_revision():
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

        return git_revision

    except subprocess.CalledProcessError:
        return "Unknown"
