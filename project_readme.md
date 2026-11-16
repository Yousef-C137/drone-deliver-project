# Autonomous Drone Delivery AI — Interactive Command Center

An interactive desktop simulation and pathfinding benchmark for autonomous drone navigation across a 10×10 urban airspace. The application models wind resistance, dynamic battery drain, no-fly zones, and mid-flight recharging stations, comparing four pathfinding algorithms in real time.

---

## Key Features

* **Interactive Map Selection:** Click directly on any open grid cell to set the **START** and **GOAL** coordinates, with automatic route recalculation.
* **Wind & Battery Physics:**
  * **Tailwinds (`0.5x`):** Reduce step cost and battery consumption.
  * **Headwinds (`2.0x` – `2.5x`):** Increase step cost and accelerate battery drain.
  * **Battery Capacity (`12.0` units):** Unconstrained algorithms crash if total energy demand exceeds battery capacity, while Battery-Aware A\* plans mandatory stops at charging stations (`+2.0` route cost penalty).
* **Live Flight Animation & Telemetry:** Step-by-step drone animation on the map, live battery depletion/recharge curve, and a real-time flight log.
* **Modular Architecture:** Strictly separates pathfinding and physics logic (`drone_engine.py`) from the Tkinter + Matplotlib GUI (`drone_ui.py`).

---

## Project Structure

```text
drone-deliver-project/
├── drone_engine.py   # 10x10 grid map, wind/battery physics, BFS, Greedy, A*, Battery-Aware A*
├── drone_ui.py       # Single-window Tkinter & Matplotlib interactive dashboard
└── README.md         # Project documentation
```

---

## Algorithms Included

| Algorithm | Optimization Goal | Wind-Aware | Battery & Recharge Aware |
| :--- | :--- | :---: | :---: |
| **BFS** | Fewest grid steps | No | No |
| **Greedy Best-First** | Closest Manhattan heuristic to goal | No | No |
| **Standard A\*** | Lowest cumulative wind cost | Yes | No |
| **Battery-Aware A\*** | Lowest wind + recharge cost within battery limits | Yes | Yes |

---

## Getting Started

### Prerequisites

* Python 3.8 or higher (`tkinter` is included with standard Python installations)
* `matplotlib`
* `numpy`

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yousef-C137/drone-deliver-project.git
   cd drone-deliver-project
   ```

2. **Install dependencies:**
   ```bash
   pip install matplotlib numpy
   ```

3. **Launch the application:**
   ```bash
   python drone_ui.py
   ```

---

## How to Use

1. **Set Coordinates:** Click any open square on the 10×10 grid to place the **START** point (green). The click mode automatically switches so your next click sets the **GOAL** point (red).
2. **Compare Algorithms:** Review the **Algorithm Benchmark** table on the right panel to see steps, route cost, energy required, explored nodes, and whether each planner achieves a safe arrival or runs out of battery.
3. **Simulate Flight:** Select an algorithm radio button and click **▶ Animate Flight** to watch the drone navigate wind zones, recharge at `CHG` stations, and update the battery telemetry graph.