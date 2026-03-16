from __future__ import annotations

import heapq
from typing import List, Tuple


def dijkstra_all_to_one(adj: List[List[Tuple[int, float]]], target: int) -> List[float]:
    """Compute shortest-path distances from all nodes to `target`.

    Works for undirected graphs when `adj` is symmetric; for directed graphs,
    you should pass a reversed adjacency list.

    Returns a list dist where dist[v] is the shortest-path distance (ms) from v to target.
    """

    n = len(adj)
    dist = [float("inf")] * n
    dist[target] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, target)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def dijkstra_one_to_one(adj: List[List[Tuple[int, float]]], source: int, target: int) -> float:
    """Shortest-path distance (ms) from `source` to `target` on a static graph."""
    n = len(adj)
    dist = [float("inf")] * n
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if u == target:
            return d_u
        if d_u != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return float("inf")
