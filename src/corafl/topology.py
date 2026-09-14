from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import stable_seed

Edge = tuple[int, int]


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def ring_matching(order: list[int], parity: int) -> list[Edge]:
    n = len(order)
    if n % 2:
        raise ValueError("Ring matching requires an even number of clients")
    if parity % 2 == 0:
        pairs = [(order[k], order[k + 1]) for k in range(0, n, 2)]
    else:
        pairs = [(order[k], order[(k + 1) % n]) for k in range(1, n, 2)]
    return sorted({canonical_edge(i, j) for i, j in pairs})


def natural_ring_order(n_clients: int) -> list[int]:
    return list(range(n_clients))


def domain_aware_ring_order(domains: np.ndarray) -> list[int]:
    return sorted(range(len(domains)), key=lambda idx: (int(domains[idx]), idx))


def safety_matching(round_id: int, domains: np.ndarray) -> list[Edge]:
    phase = round_id % 4
    if phase == 0:
        return ring_matching(domain_aware_ring_order(domains), 0)
    if phase == 2:
        return ring_matching(domain_aware_ring_order(domains), 1)
    return []


def random_matching(active: np.ndarray, seed: int, round_id: int, label: str) -> list[Edge]:
    nodes = np.flatnonzero(active)
    rng = np.random.default_rng(stable_seed(label, seed, round_id))
    nodes = rng.permutation(nodes)
    return [canonical_edge(int(nodes[k]), int(nodes[k + 1])) for k in range(0, len(nodes) - 1, 2)]


def exponential_matching(n_clients: int, round_id: int) -> list[Edge]:
    bit_count = int(np.log2(n_clients))
    partner_bit = 1 << (round_id % bit_count)
    return [canonical_edge(i, i ^ partner_bit) for i in range(n_clients) if i < (i ^ partner_bit)]


def static_random_matchings(n_clients: int, seed: int, count: int = 4) -> list[list[Edge]]:
    rng = np.random.default_rng(stable_seed("static-rr", seed, n_clients))
    matchings: list[list[Edge]] = []
    seen: set[tuple[Edge, ...]] = set()
    while len(matchings) < count:
        order = rng.permutation(n_clients)
        edges = tuple(sorted(canonical_edge(int(order[k]), int(order[k + 1])) for k in range(0, n_clients, 2)))
        if edges in seen:
            continue
        seen.add(edges)
        matchings.append(list(edges))
    return matchings


def complete_graph_factorization(n_clients: int) -> list[list[Edge]]:
    """Return a one-factorization of K_n using the circle construction."""
    if n_clients < 2 or n_clients % 2:
        raise ValueError("Complete-graph factorization requires an even number of clients")
    rotating = list(range(n_clients))
    factors: list[list[Edge]] = []
    for _ in range(n_clients - 1):
        factors.append(
            sorted(
                canonical_edge(rotating[k], rotating[-1 - k])
                for k in range(n_clients // 2)
            )
        )
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    if len({edge for factor in factors for edge in factor}) != n_clients * (n_clients - 1) // 2:
        raise RuntimeError("Invalid one-factorization")
    return factors


def matcha_matching(n_clients: int, seed: int, round_id: int) -> list[Edge]:
    """Sample one complete-graph matching with the symmetric MATCHA budget.

    For an equal-cost complete base graph, symmetry gives equal activation
    probabilities. MATCHA explicitly permits selecting one matching per round;
    this implementation samples one factor independently and deterministically
    from the run seed.
    """
    factors = complete_graph_factorization(n_clients)
    rng = np.random.default_rng(stable_seed("matcha", seed, round_id, n_clients))
    return factors[int(rng.integers(0, len(factors)))]


def greedy_pairwise_matching(score: np.ndarray, active: np.ndarray) -> list[Edge]:
    candidates: list[tuple[float, int, int]] = []
    nodes = np.flatnonzero(active)
    for pos, i_raw in enumerate(nodes):
        i = int(i_raw)
        for j_raw in nodes[pos + 1 :]:
            j = int(j_raw)
            candidates.append((float(score[i, j]), i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched: set[int] = set()
    result: list[Edge] = []
    for _, i, j in candidates:
        if i not in matched and j not in matched:
            matched.add(i)
            matched.add(j)
            result.append((i, j))
    return result


def filter_edges(edges: list[Edge], active: np.ndarray, link_up: np.ndarray) -> tuple[list[Edge], list[Edge]]:
    attempted = [(i, j) for i, j in edges if active[i] and active[j]]
    successful = [(i, j) for i, j in attempted if link_up[i, j]]
    return attempted, successful


def mixing_matrix(n_clients: int, edges: list[Edge], active: np.ndarray) -> np.ndarray:
    matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
    degrees = np.zeros(n_clients, dtype=np.int64)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    for i, j in edges:
        weight = 1.0 / (1.0 + max(int(degrees[i]), int(degrees[j])))
        matrix[i, j] = weight
        matrix[j, i] = weight
    for i in range(n_clients):
        matrix[i, i] = 1.0 - matrix[i].sum() if active[i] else 1.0
    return matrix


def component_count(n_clients: int, edges: list[Edge], active: np.ndarray) -> tuple[int, float]:
    nodes = set(int(x) for x in np.flatnonzero(active))
    if not nodes:
        return 0, 0.0
    adjacency = {node: set() for node in nodes}
    for i, j in edges:
        if i in nodes and j in nodes:
            adjacency[i].add(j)
            adjacency[j].add(i)
    sizes: list[int] = []
    while nodes:
        root = nodes.pop()
        stack = [root]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in nodes:
                    nodes.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return len(sizes), max(sizes) / max(sum(sizes), 1)


def _connected(nodes: set[int], edges: set[Edge]) -> bool:
    if len(nodes) <= 1:
        return True
    adjacency = {node: set() for node in nodes}
    for i, j in edges:
        if i in nodes and j in nodes:
            adjacency[i].add(j)
            adjacency[j].add(i)
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == nodes


def verify_one_domain_certificate(domains: np.ndarray, period: int = 4, window: int = 4) -> dict[str, object]:
    n_clients = len(domains)
    failures: list[dict[str, object]] = []
    for failed_domain in sorted(set(int(x) for x in domains)):
        surviving = {i for i in range(n_clients) if int(domains[i]) != failed_domain}
        for start in range(period):
            union: set[Edge] = set()
            for offset in range(window):
                union.update(safety_matching((start + offset) % period, domains))
            union = {edge for edge in union if edge[0] in surviving and edge[1] in surviving}
            if not _connected(surviving, union):
                failures.append({"failed_domain": failed_domain, "window_start": start})
    return {"valid": not failures, "failure_model": "any one declared domain removed", "period": period, "window": window, "n_clients": n_clients, "n_domains": len(set(int(x) for x in domains)), "failures": failures}


@dataclass
class TopologyState:
    neighbor_ema: np.ndarray
    reliability: np.ndarray
    static_matchings: list[list[Edge]]


def initialize_state(class_histograms: np.ndarray, seed: int) -> TopologyState:
    n_clients = len(class_histograms)
    return TopologyState(class_histograms.copy(), np.ones((n_clients, n_clients), dtype=np.float64), static_random_matchings(n_clients, seed))


def _similarity_scores(histograms: np.ndarray) -> np.ndarray:
    n_clients = len(histograms)
    scores = np.full((n_clients, n_clients), -np.inf, dtype=np.float64)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            value = -float(np.abs(histograms[i] - histograms[j]).sum())
            scores[i, j] = scores[j, i] = value
    return scores


def _cora_scores(histograms: np.ndarray, domains: np.ndarray, state: TopologyState, coverage_weight: float, domain_weight: float, reliability_weight: float) -> np.ndarray:
    n_clients, n_classes = histograms.shape
    target = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    debt = np.maximum(target[None, :] - state.neighbor_ema, 0.0)
    scores = np.full((n_clients, n_clients), -np.inf, dtype=np.float64)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            coverage = float(np.dot(debt[i], histograms[j]) + np.dot(debt[j], histograms[i]))
            value = coverage_weight * coverage + domain_weight * float(domains[i] != domains[j]) + reliability_weight * float(state.reliability[i, j])
            scores[i, j] = scores[j, i] = value
    return scores


def proposed_edges(method: str, round_id: int, active: np.ndarray, histograms: np.ndarray, domains: np.ndarray, state: TopologyState, seed: int, coverage_weight: float, domain_weight: float, reliability_weight: float) -> tuple[list[Edge], bool]:
    n_clients = len(active)
    adaptive = False
    if method == "ring_gt":
        edges = ring_matching(natural_ring_order(n_clients), round_id % 2)
    elif method == "random_gossip_gt":
        edges = random_matching(active, seed, round_id, "random-gossip")
    elif method == "exponential_gt":
        edges = exponential_matching(n_clients, round_id)
    elif method == "static_rr_gt":
        edges = state.static_matchings[round_id % len(state.static_matchings)]
    elif method == "matcha_gt":
        edges = matcha_matching(n_clients, seed, round_id)
    elif method == "adaptive_similarity_gt":
        edges = greedy_pairwise_matching(_similarity_scores(histograms), active)
        adaptive = True
    elif method in {"cora_fl_v1", "cora_fl"}:
        safety = safety_matching(round_id, domains)
        if safety:
            edges = safety
        else:
            edges = greedy_pairwise_matching(_cora_scores(histograms, domains, state, coverage_weight, domain_weight, reliability_weight), active)
            adaptive = True
    else:
        raise ValueError(f"No peer topology for method: {method}")
    return edges, adaptive


def update_adaptive_state(state: TopologyState, attempted: list[Edge], successful: list[Edge], histograms: np.ndarray, ema_rate: float, reliability_ema_rate: float) -> None:
    successful_set = set(successful)
    for i, j in attempted:
        outcome = 1.0 if (i, j) in successful_set else 0.0
        updated = (1.0 - reliability_ema_rate) * state.reliability[i, j] + reliability_ema_rate * outcome
        state.reliability[i, j] = state.reliability[j, i] = updated
    for i, j in successful:
        state.neighbor_ema[i] = (1.0 - ema_rate) * state.neighbor_ema[i] + ema_rate * histograms[j]
        state.neighbor_ema[j] = (1.0 - ema_rate) * state.neighbor_ema[j] + ema_rate * histograms[i]


def cross_domain_fraction(edges: list[Edge], domains: np.ndarray) -> float:
    if not edges:
        return 0.0
    return float(np.mean([domains[i] != domains[j] for i, j in edges]))
