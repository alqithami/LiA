from __future__ import annotations

import heapq
from typing import Callable, List, Optional, Tuple

# Signature: (u, v, departure_time_ms, base_weight_ms) -> travel_time_ms
EdgeWeightFn = Callable[[int, int, float, float], float]


def earliest_arrival_time(
    adj: List[List[Tuple[int, float]]],
    source: int,
    target: int,
    departure_time_ms: float,
    weight_fn: Optional[EdgeWeightFn] = None,
) -> float:
    """Earliest-arrival time on a time-dependent (FIFO) graph.

    If weight_fn is None, edges are treated as static.

    This is a Dijkstra-style label-setting algorithm valid for FIFO travel-time functions
    where departing later on an edge cannot result in an earlier arrival than departing earlier.

    Returns the earliest arrival time (ms since t=0) at `target`.
    """

    if weight_fn is None:
        weight_fn = lambda u, v, t, w: w  # type: ignore

    n = len(adj)
    arr = [float("inf")] * n
    arr[source] = float(departure_time_ms)

    pq: List[Tuple[float, int]] = [(arr[source], source)]

    while pq:
        t_u, u = heapq.heappop(pq)
        if t_u != arr[u]:
            continue
        if u == target:
            return t_u
        for v, w in adj[u]:
            travel = float(weight_fn(u, v, t_u, w))
            t_v = t_u + travel
            if t_v < arr[v]:
                arr[v] = t_v
                heapq.heappush(pq, (t_v, v))

    return float("inf")
