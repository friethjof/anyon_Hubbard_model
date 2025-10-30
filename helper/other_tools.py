def time_str(sec: float) -> str:
    """
    Convert time in seconds to a human-readable string.

    Args:
        sec (float): Time duration in seconds.

    Returns:
        str: Formatted string as 'Hh:Mm:Ss' where H = hours,
             M = minutes, and S = seconds with two decimal places.

    Example:
        >>> time_str(3661.23)
        '1h:1m:1.23s'
    """
    hour = sec // 3600
    sec %= 3600
    minute = sec // 60
    sec %= 60
    return f"{int(hour)}h:{int(minute)}m:{sec:.2f}s"
