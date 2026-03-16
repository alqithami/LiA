from __future__ import annotations

from typing import Iterable, List, Tuple


def clique_number(intervals: Iterable[Tuple[float, float]]) -> int:
    """Maximum number of intervals overlapping at any time.

    Intervals are treated as closed on the left and open on the right: [start, end).
    """

    events: List[Tuple[float, int]] = []
    for a, b in intervals:
        events.append((float(a), +1))
        events.append((float(b), -1))

    # Sort by time; ties resolved by ends (-1) before starts (+1) for [start, end) convention
    events.sort(key=lambda x: (x[0], x[1]))

    count = 0
    max_count = 0
    for _, delta in events:
        count += delta
        if count > max_count:
            max_count = count
    return int(max_count)
