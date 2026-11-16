import heapq
import time
from collections import deque

# =========================================================
# 1. 10x10 CITY AIRSPACE, WIND ZONES & CHARGING STATIONS
# =========================================================
# 0 = Open Air, 1 = No-Fly Zone (Building)
GRID = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
]

GRID_SIZE = len(GRID)
RECHARGE_STATIONS = [(4, 4), (2, 7), (7, 2)]

# Wind multipliers: < 1.0 = Tailwind, > 1.0 = Headwind
NORMAL_COST = 1.0
WIND_COSTS = {
    (0, 1): 0.5, (0, 2): 0.5,
    (8, 5): 0.5, (8, 6): 0.5,
    (9, 7): 0.5, (9, 8): 0.5,
    (2, 2): 2.5, (2, 3): 2.5, (2, 4): 2.0,
    (3, 5): 2.5, (4, 5): 2.5,
    (5, 7): 2.0, (6, 7): 2.5, (7, 7): 2.0,
}

# Battery & Physics Settings
BATTERY_CAPACITY = 12.0
INITIAL_BATTERY = 12.0
RECHARGE_PENALTY = 2.0
MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def is_valid(pos):
    r, c = pos
    if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
        return False
    return GRID[r][c] == 0


def get_step_cost(pos):
    return WIND_COSTS.get(pos, NORMAL_COST)


def get_battery_drain(pos):
    """Battery drain scales directly with wind resistance."""
    return get_step_cost(pos)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def evaluate_unconstrained_path(path):
    if not path:
        return float("inf"), float("inf")
    cost = sum(get_step_cost(path[i]) for i in range(1, len(path)))
    energy = sum(get_battery_drain(path[i]) for i in range(1, len(path)))
    return round(cost, 2), round(energy, 2)


# =========================================================
# 2. PATHFINDING ALGORITHMS
# =========================================================
def bfs(start, goal):
    t0 = time.perf_counter()
    queue = deque([[start]])
    visited = {start}
    explored = 0

    while queue:
        path = queue.popleft()
        current = path[-1]
        explored += 1
        if current == goal:
            return path, explored, round((time.perf_counter() - t0) * 1000, 3)

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)
            if is_valid(nxt) and nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])

    return None, explored, round((time.perf_counter() - t0) * 1000, 3)


def greedy(start, goal):
    t0 = time.perf_counter()
    pq = [(heuristic(start, goal), [start])]
    visited = {start}
    explored = 0

    while pq:
        _, path = heapq.heappop(pq)
        current = path[-1]
        explored += 1
        if current == goal:
            return path, explored, round((time.perf_counter() - t0) * 1000, 3)

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)
            if is_valid(nxt) and nxt not in visited:
                visited.add(nxt)
                heapq.heappush(pq, (heuristic(nxt, goal), path + [nxt]))

    return None, explored, round((time.perf_counter() - t0) * 1000, 3)


def astar(start, goal):
    t0 = time.perf_counter()
    pq = [(0.0, 0.0, start)]
    came_from = {}
    g_cost = {start: 0.0}
    explored = 0

    while pq:
        _, curr_cost, current = heapq.heappop(pq)
        if curr_cost != g_cost.get(current):
            continue
        explored += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, round(curr_cost, 2), explored, round((time.perf_counter() - t0) * 1000, 3)

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)
            if not is_valid(nxt):
                continue
            new_cost = curr_cost + get_step_cost(nxt)
            if new_cost < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = new_cost
                came_from[nxt] = current
                h = 0.5 * heuristic(nxt, goal)
                heapq.heappush(pq, (new_cost + h, new_cost, nxt))

    return None, float("inf"), explored, round((time.perf_counter() - t0) * 1000, 3)


def astar_with_battery(start, goal):
    t0 = time.perf_counter()
    start_state = (start, INITIAL_BATTERY)
    pq = [(0.0, 0.0, start_state)]
    g_cost = {start_state: 0.0}
    came_from = {}
    explored = 0

    while pq:
        _, curr_cost, state = heapq.heappop(pq)
        if curr_cost != g_cost.get(state):
            continue
        explored += 1
        pos, bat = state

        if pos == goal:
            path = [state]
            while state in came_from:
                state = came_from[state]
                path.append(state)
            path.reverse()
            return path, round(curr_cost, 2), explored, round((time.perf_counter() - t0) * 1000, 3)

        # Option 1: Move to neighbor (battery drain scales with wind)
        for dr, dc in MOVES:
            nxt = (pos[0] + dr, pos[1] + dc)
            if not is_valid(nxt):
                continue
            drain = get_battery_drain(nxt)
            if bat < drain:
                continue
            new_bat = round(bat - drain, 2)
            new_state = (nxt, new_bat)
            new_cost = curr_cost + get_step_cost(nxt)
            if new_cost < g_cost.get(new_state, float("inf")):
                g_cost[new_state] = new_cost
                came_from[new_state] = state
                h = 0.5 * heuristic(nxt, goal)
                heapq.heappush(pq, (new_cost + h, new_cost, new_state))

        # Option 2: Recharge at station (adds RECHARGE_PENALTY to route cost)
        if pos in RECHARGE_STATIONS and bat < BATTERY_CAPACITY:
            new_state = (pos, BATTERY_CAPACITY)
            new_cost = curr_cost + RECHARGE_PENALTY
            if new_cost < g_cost.get(new_state, float("inf")):
                g_cost[new_state] = new_cost
                came_from[new_state] = state
                h = 0.5 * heuristic(pos, goal)
                heapq.heappush(pq, (new_cost + h, new_cost, new_state))

    return None, float("inf"), explored, round((time.perf_counter() - t0) * 1000, 3)


def run_all_planners(start, goal):
    """Runs all four algorithms and returns structured metrics for the UI."""
    bfs_path, bfs_exp, bfs_ms = bfs(start, goal)
    bfs_cost, bfs_energy = evaluate_unconstrained_path(bfs_path)

    greedy_path, greedy_exp, greedy_ms = greedy(start, goal)
    greedy_cost, greedy_energy = evaluate_unconstrained_path(greedy_path)

    astar_path, astar_cost, astar_exp, astar_ms = astar(start, goal)
    _, astar_energy = evaluate_unconstrained_path(astar_path)

    bat_path, bat_cost, bat_exp, bat_ms = astar_with_battery(start, goal)
    if bat_path:
        bat_steps = sum(1 for i in range(1, len(bat_path)) if bat_path[i][0] != bat_path[i - 1][0])
        bat_energy = round(sum(max(0.0, bat_path[i - 1][1] - bat_path[i][1]) for i in range(1, len(bat_path))), 2)
        bat_recharges = sum(1 for i in range(1, len(bat_path)) if bat_path[i][1] > bat_path[i - 1][1])
    else:
        bat_steps = 0
        bat_energy = float("inf")
        bat_recharges = 0

    return {
        "BFS": {
            "path": bfs_path,
            "steps": len(bfs_path) - 1 if bfs_path else 0,
            "cost": bfs_cost,
            "energy": bfs_energy,
            "recharges": 0,
            "explored": bfs_exp,
            "time_ms": bfs_ms,
            "feasible": bfs_energy <= BATTERY_CAPACITY,
        },
        "Greedy": {
            "path": greedy_path,
            "steps": len(greedy_path) - 1 if greedy_path else 0,
            "cost": greedy_cost,
            "energy": greedy_energy,
            "recharges": 0,
            "explored": greedy_exp,
            "time_ms": greedy_ms,
            "feasible": greedy_energy <= BATTERY_CAPACITY,
        },
        "Standard A*": {
            "path": astar_path,
            "steps": len(astar_path) - 1 if astar_path else 0,
            "cost": astar_cost,
            "energy": astar_energy,
            "recharges": 0,
            "explored": astar_exp,
            "time_ms": astar_ms,
            "feasible": astar_energy <= BATTERY_CAPACITY,
        },
        "Battery A*": {
            "path": bat_path,
            "steps": bat_steps,
            "cost": bat_cost,
            "energy": bat_energy,
            "recharges": bat_recharges,
            "explored": bat_exp,
            "time_ms": bat_ms,
            "feasible": bat_path is not None,
        },
    }
