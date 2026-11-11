"""One way to print a recording time, for every surface that prints one.

Four modules each carried a private, identical ``_fmt_time``; the player's own readout is the
fifth reader, which is the point at which a copy becomes a mechanism.

Milliseconds, not tenths: two frames of 30 fps video are 33 ms apart, so a readout in tenths
shows the same time for adjacent frames and cannot be used to say which one you are on.
"""

#: The widest `format_time` renders below a thousand minutes — what a readout reserves, so the
#: controls beside it do not move as the clock runs. Three digits of minutes is a 16-hour
#: recording.
TIME_WIDEST = "999:59.999"


def format_time(t_ns: int) -> str:
    """``m:ss.mmm`` for a nanosecond timestamp."""
    seconds = t_ns / 1e9
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"
