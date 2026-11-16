import tkinter as tk
from tkinter import ttk

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# Import algorithms, grid data, and physics from the separate engine file
from drone_engine import (
    BATTERY_CAPACITY,
    GRID,
    GRID_SIZE,
    RECHARGE_PENALTY,
    RECHARGE_STATIONS,
    WIND_COSTS,
    is_valid,
    run_all_planners,
)

# =========================================================
# THEME & STYLING PALETTE
# =========================================================
THEME = {
    "bg_main": "#0f172a",
    "bg_panel": "#1e293b",
    "bg_card": "#0b1120",
    "border": "#334155",
    "text_primary": "#f8fafc",
    "text_muted": "#94a3b8",
    "accent_blue": "#38bdf8",
    "accent_amber": "#f59e0b",
    "accent_green": "#22c55e",
    "accent_red": "#ef4444",
    "cell_calm": "#1e293b",
    "cell_nofly": "#020617",
    "cell_tailwind": "#134e4a",
    "cell_headwind": "#7f1d1d",
    "cell_station": "#0284c7",
    "cell_start": "#16a34a",
    "cell_goal": "#dc2626",
}


class DroneDeliveryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Delivery AI — Interactive Command Center")
        self.root.geometry("1380x840")
        self.root.configure(bg=THEME["bg_main"])

        # Interactive state
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)
        self.click_mode = tk.StringVar(value="START")
        self.selected_algo = tk.StringVar(value="Battery A*")
        self.results = {}
        self.sim_step = 0
        self.animating = False

        self._configure_ttk_styles()
        self._build_layout()
        self._recompute_routes()

    def _configure_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=THEME["bg_main"])
        style.configure("Panel.TFrame", background=THEME["bg_panel"])
        style.configure("Title.TLabel", background=THEME["bg_main"], foreground=THEME["text_primary"], font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=THEME["bg_main"], foreground=THEME["text_muted"], font=("Segoe UI", 10))
        style.configure("PanelHeader.TLabel", background=THEME["bg_panel"], foreground=THEME["accent_blue"], font=("Segoe UI", 11, "bold"))
        style.configure("PanelText.TLabel", background=THEME["bg_panel"], foreground=THEME["text_primary"], font=("Segoe UI", 10))
        style.configure("Treeview", background=THEME["bg_card"], foreground=THEME["text_primary"], fieldbackground=THEME["bg_card"], rowheight=26, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=THEME["border"], foreground=THEME["text_primary"], font=("Segoe UI", 9, "bold"))

    def _build_layout(self):
        # Top Header Bar
        header = ttk.Frame(self.root, padding=(18, 12, 18, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="AUTONOMOUS DRONE DELIVERY AI — INTERACTIVE MAP & TELEMETRY", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Click any open square on the map to set START or GOAL • Wind resistance dynamically scales battery drain",
            style="Sub.TLabel",
        ).pack(anchor=tk.W)

        # Main Two-Column Body
        body = ttk.Frame(self.root, padding=(18, 6, 18, 16))
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Interactive Map Canvas
        map_container = tk.Frame(body, bg=THEME["bg_panel"], highlightbackground=THEME["border"], highlightthickness=1)
        map_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.fig = Figure(figsize=(7.4, 7.2), dpi=100, facecolor=THEME["bg_panel"])
        self.ax_map = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

        # Right Column: Controls, Comparison Table, Battery Graph & Log
        right_panel = tk.Frame(body, bg=THEME["bg_panel"], width=520, highlightbackground=THEME["border"], highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        self._build_controls_section(right_panel)
        self._build_comparison_table(right_panel)
        self._build_battery_chart(right_panel)
        self._build_telemetry_log(right_panel)

    def _build_controls_section(self, parent):
        box = tk.Frame(parent, bg=THEME["bg_panel"], padx=14, pady=10)
        box.pack(fill=tk.X)

        tk.Label(box, text="1. MAP CLICK MODE & ALGORITHM", bg=THEME["bg_panel"], fg=THEME["accent_blue"], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        # Click Mode Radio Buttons
        mode_row = tk.Frame(box, bg=THEME["bg_panel"], pady=6)
        mode_row.pack(fill=tk.X)

        tk.Radiobutton(
            mode_row, text="Set START (Green)", variable=self.click_mode, value="START",
            bg=THEME["bg_panel"], fg=THEME["accent_green"], selectcolor=THEME["bg_card"],
            activebackground=THEME["bg_panel"], font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 14))

        tk.Radiobutton(
            mode_row, text="Set GOAL (Red)", variable=self.click_mode, value="GOAL",
            bg=THEME["bg_panel"], fg=THEME["accent_red"], selectcolor=THEME["bg_card"],
            activebackground=THEME["bg_panel"], font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT)

        self.coord_label = tk.Label(
            box,
            text=f"Start: {self.start_pos}   |   Goal: {self.goal_pos}   |   Max Battery: {BATTERY_CAPACITY} units",
            bg=THEME["bg_card"], fg=THEME["text_primary"], font=("Consolas", 10, "bold"), pady=6, padx=8
        )
        self.coord_label.pack(fill=tk.X, pady=(2, 8))

        # Algorithm Selector Buttons
        algo_row = tk.Frame(box, bg=THEME["bg_panel"])
        algo_row.pack(fill=tk.X)
        for algo in ["BFS", "Greedy", "Standard A*", "Battery A*"]:
            tk.Radiobutton(
                algo_row, text=algo, variable=self.selected_algo, value=algo,
                command=self._on_algo_change,
                bg=THEME["bg_panel"], fg=THEME["text_primary"], selectcolor=THEME["bg_card"],
                activebackground=THEME["bg_panel"], font=("Segoe UI", 9, "bold")
            ).pack(side=tk.LEFT, padx=(0, 10))

        # Action Buttons (Animate / Reset)
        btn_row = tk.Frame(box, bg=THEME["bg_panel"], pady=6)
        btn_row.pack(fill=tk.X)

        self.play_btn = tk.Button(
            btn_row, text="▶ Animate Flight", command=self._toggle_animation,
            bg=THEME["accent_blue"], fg="#0f172a", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=12, pady=5, cursor="hand2"
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row, text="↺ Reset (0,0) → (9,9)", command=self._reset_defaults,
            bg=THEME["border"], fg=THEME["text_primary"], font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=12, pady=5, cursor="hand2"
        ).pack(side=tk.LEFT)

        self.status_banner = tk.Label(
            box, text="Ready", bg=THEME["bg_card"], fg=THEME["accent_amber"],
            font=("Segoe UI", 9, "bold"), anchor="w", padx=8, pady=5
        )
        self.status_banner.pack(fill=tk.X, pady=(4, 0))

    def _build_comparison_table(self, parent):
        box = tk.Frame(parent, bg=THEME["bg_panel"], padx=14, pady=6)
        box.pack(fill=tk.X)

        tk.Label(box, text="2. ALGORITHM BENCHMARK", bg=THEME["bg_panel"], fg=THEME["accent_blue"], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

        cols = ("algo", "steps", "cost", "energy", "explored", "status")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=4)
        headings = {
            "algo": ("Algorithm", 105),
            "steps": ("Steps", 50),
            "cost": ("Cost", 55),
            "energy": ("Energy", 60),
            "explored": ("Nodes", 60),
            "status": ("Battery Check", 130),
        }
        for col, (title, width) in headings.items():
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)

    def _build_battery_chart(self, parent):
        box = tk.Frame(parent, bg=THEME["bg_panel"], padx=14, pady=6)
        box.pack(fill=tk.X)

        tk.Label(box, text="3. LIVE BATTERY TELEMETRY", bg=THEME["bg_panel"], fg=THEME["accent_blue"], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        self.bat_fig = Figure(figsize=(4.8, 2.0), dpi=95, facecolor=THEME["bg_panel"])
        self.ax_bat = self.bat_fig.add_subplot(111)
        self.bat_canvas = FigureCanvasTkAgg(self.bat_fig, master=box)
        self.bat_canvas.get_tk_widget().pack(fill=tk.X, pady=(4, 0))

    def _build_telemetry_log(self, parent):
        box = tk.Frame(parent, bg=THEME["bg_panel"], padx=14, pady=6)
        box.pack(fill=tk.BOTH, expand=True)

        tk.Label(box, text="4. FLIGHT STEP LOG", bg=THEME["bg_panel"], fg=THEME["accent_blue"], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        self.log_text = tk.Text(
            box, height=8, bg=THEME["bg_card"], fg=THEME["text_primary"],
            font=("Consolas", 9), relief=tk.FLAT, padx=8, pady=6, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    # =========================================================
    # EVENT HANDLERS & SIMULATION LOGIC
    # =========================================================
    def _on_map_click(self, event):
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        clicked = (row, col)

        if not is_valid(clicked):
            self.status_banner.config(text=f"Blocked cell {clicked}! Choose an open square.", fg=THEME["accent_red"])
            return

        if self.click_mode.get() == "START":
            if clicked == self.goal_pos:
                self.status_banner.config(text="Start cannot overlap Goal!", fg=THEME["accent_red"])
                return
            self.start_pos = clicked
            # Automatically switch to GOAL after picking START for smooth UX
            self.click_mode.set("GOAL")
        else:
            if clicked == self.start_pos:
                self.status_banner.config(text="Goal cannot overlap Start!", fg=THEME["accent_red"])
                return
            self.goal_pos = clicked
            self.click_mode.set("START")

        self.animating = False
        self.play_btn.config(text="▶ Animate Flight")
        self._recompute_routes()

    def _on_algo_change(self):
        self.animating = False
        self.play_btn.config(text="▶ Animate Flight")
        path = self.results[self.selected_algo.get()]["path"]
        self.sim_step = len(path) - 1 if path else 0
        self._refresh_visuals()

    def _reset_defaults(self):
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)
        self.click_mode.set("START")
        self.selected_algo.set("Battery A*")
        self.animating = False
        self.play_btn.config(text="▶ Animate Flight")
        self._recompute_routes()

    def _recompute_routes(self):
        self.coord_label.config(
            text=f"Start: {self.start_pos}   |   Goal: {self.goal_pos}   |   Max Battery: {BATTERY_CAPACITY} units"
        )
        self.results = run_all_planners(self.start_pos, self.goal_pos)

        # Update comparison table
        for item in self.tree.get_children():
            self.tree.delete(item)

        for name, data in self.results.items():
            status = "SAFE ARRIVAL" if data["feasible"] else "CRASH (Low Bat)"
            if not data["path"]:
                status = "NO ROUTE"
            self.tree.insert(
                "", tk.END,
                values=(name, data["steps"], data["cost"], data["energy"], data["explored"], status)
            )

        active_path = self.results[self.selected_algo.get()]["path"]
        self.sim_step = len(active_path) - 1 if active_path else 0
        self._refresh_visuals()

    def _toggle_animation(self):
        path = self.results[self.selected_algo.get()]["path"]
        if not path:
            return
        if self.animating:
            self.animating = False
            self.play_btn.config(text="▶ Animate Flight")
        else:
            self.animating = True
            self.play_btn.config(text="⏸ Pause")
            if self.sim_step >= len(path) - 1:
                self.sim_step = 0
            self._step_animation()

    def _step_animation(self):
        if not self.animating:
            return
        path = self.results[self.selected_algo.get()]["path"]
        if not path:
            self.animating = False
            return

        self._refresh_visuals()
        if self.sim_step < len(path) - 1:
            self.sim_step += 1
            self.root.after(260, self._step_animation)
        else:
            self.animating = False
            self.play_btn.config(text="▶ Replay Flight")

    # =========================================================
    # RENDERING: MAP, BATTERY CHART & LOG
    # =========================================================
    def _refresh_visuals(self):
        algo_name = self.selected_algo.get()
        data = self.results[algo_name]
        path = data["path"]

        # Status banner
        if not path:
            self.status_banner.config(text=f"{algo_name}: No valid path found!", fg=THEME["accent_red"])
        elif data["feasible"]:
            self.status_banner.config(
                text=f"{algo_name}: Safe flight • Cost {data['cost']} • Energy {data['energy']} • Recharges: {data['recharges']}",
                fg=THEME["accent_green"],
            )
        else:
            self.status_banner.config(
                text=f"WARNING: {algo_name} requires {data['energy']} energy (exceeds {BATTERY_CAPACITY} battery cap!)",
                fg=THEME["accent_red"],
            )

        self._draw_map(algo_name, path)
        self._draw_battery_profile(algo_name, path)
        self._update_log(algo_name, path)

    def _draw_map(self, algo_name, path):
        self.ax_map.clear()
        self.ax_map.set_facecolor(THEME["bg_panel"])

        visual = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if GRID[r][c] == 1:
                    visual[r, c] = 1
                elif (r, c) in WIND_COSTS:
                    visual[r, c] = 2 if WIND_COSTS[(r, c)] < 1.0 else 3

        for r, c in RECHARGE_STATIONS:
            visual[r, c] = 4
        visual[self.start_pos] = 5
        visual[self.goal_pos] = 6

        palette = [
            THEME["cell_calm"],
            THEME["cell_nofly"],
            THEME["cell_tailwind"],
            THEME["cell_headwind"],
            THEME["cell_station"],
            THEME["cell_start"],
            THEME["cell_goal"],
        ]
        self.ax_map.imshow(visual, cmap=ListedColormap(palette), vmin=0, vmax=6)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                if GRID[r][c] == 1:
                    self.ax_map.text(c, r, "X", ha="center", va="center", color="#475569", fontsize=11, fontweight="bold")
                elif pos == self.start_pos:
                    self.ax_map.text(c, r, "START", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                elif pos == self.goal_pos:
                    self.ax_map.text(c, r, "GOAL", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                elif pos in RECHARGE_STATIONS:
                    self.ax_map.text(c, r, "CHG", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                elif pos in WIND_COSTS:
                    w = WIND_COSTS[pos]
                    lbl_color = "#5eead4" if w < 1.0 else "#fca5a5"
                    self.ax_map.text(c, r, f"{w}x", ha="center", va="center", color=lbl_color, fontsize=8.5, fontweight="bold")

        if path:
            visible_path = path[: self.sim_step + 1]
            coords = [step[0] if isinstance(step[0], tuple) else step for step in visible_path]
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            self.ax_map.plot(cols, rows, color=THEME["accent_amber"], linewidth=3.2, marker="o", markersize=5, zorder=4)

            # Current drone position marker
            cur_r, cur_c = coords[-1]
            self.ax_map.scatter([cur_c], [cur_r], s=260, color="#facc15", edgecolors="#0f172a", linewidth=2, marker="D", zorder=6)

            # Highlight recharge stops
            if algo_name == "Battery A*":
                for i in range(1, len(visible_path)):
                    if visible_path[i][1] > visible_path[i - 1][1]:
                        rr, cc = visible_path[i][0]
                        self.ax_map.scatter([cc], [rr], s=320, facecolors="none", edgecolors=THEME["accent_blue"], linewidth=3, zorder=5)

        self.ax_map.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        self.ax_map.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        self.ax_map.grid(which="minor", color=THEME["border"], linestyle="-", linewidth=1)
        self.ax_map.tick_params(which="minor", bottom=False, left=False)
        self.ax_map.set_xticks(range(GRID_SIZE))
        self.ax_map.set_yticks(range(GRID_SIZE))
        self.ax_map.tick_params(colors=THEME["text_muted"])
        self.ax_map.set_title(f"{algo_name} Route ({self.start_pos} → {self.goal_pos})", color=THEME["text_primary"], fontsize=12, fontweight="bold", pad=10)

        legend_items = [
            mpatches.Patch(color=THEME["cell_start"], label="Start"),
            mpatches.Patch(color=THEME["cell_goal"], label="Goal"),
            mpatches.Patch(color=THEME["cell_station"], label="Recharge"),
            mpatches.Patch(color=THEME["cell_tailwind"], label="Tailwind (0.5x)"),
            mpatches.Patch(color=THEME["cell_headwind"], label="Headwind (2-2.5x)"),
            mpatches.Patch(color=THEME["cell_nofly"], label="No-Fly"),
        ]
        self.ax_map.legend(
            handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.06),
            ncol=3, frameon=True, facecolor=THEME["bg_card"], edgecolor=THEME["border"],
            labelcolor=THEME["text_primary"], fontsize=8.5
        )
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_battery_profile(self, algo_name, path):
        self.ax_bat.clear()
        self.ax_bat.set_facecolor(THEME["bg_card"])

        if path:
            visible = path[: self.sim_step + 1]
            if algo_name == "Battery A*":
                levels = [b for _, b in visible]
            else:
                from drone_engine import get_battery_drain
                bat = BATTERY_CAPACITY
                levels = [bat]
                for i in range(1, len(visible)):
                    bat = round(bat - get_battery_drain(visible[i]), 2)
                    levels.append(bat)

            xs = list(range(len(levels)))
            self.ax_bat.plot(xs, levels, color=THEME["accent_blue"], linewidth=2.2, marker="o", markersize=3.5)
            self.ax_bat.fill_between(xs, levels, color=THEME["accent_blue"], alpha=0.15)

        self.ax_bat.axhline(0, color=THEME["accent_red"], linestyle="--", linewidth=1)
        self.ax_bat.axhline(BATTERY_CAPACITY, color=THEME["accent_green"], linestyle="--", linewidth=0.9)
        self.ax_bat.set_ylim(min(-4, -1), BATTERY_CAPACITY + 1.5)
        self.ax_bat.set_ylabel("Battery", color=THEME["text_muted"], fontsize=8.5)
        self.ax_bat.tick_params(colors=THEME["text_muted"], labelsize=8)
        self.ax_bat.grid(True, color=THEME["border"], alpha=0.5)
        self.bat_fig.tight_layout()
        self.bat_canvas.draw()

    def _update_log(self, algo_name, path):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)

        if not path:
            self.log_text.insert(tk.END, "No feasible route available between selected points.\n")
        elif algo_name == "Battery A*":
            for i, (pos, bat) in enumerate(path[: self.sim_step + 1]):
                if i == 0:
                    line = f"Step {i:02d}: Takeoff {pos} | Bat: {bat:.1f}/{BATTERY_CAPACITY:.1f}\n"
                elif bat > path[i - 1][1]:
                    line = f"Step {i:02d}: >>> RECHARGED at {pos} -> {bat:.1f}/{BATTERY_CAPACITY:.1f} (+{RECHARGE_PENALTY} cost)\n"
                else:
                    drain = round(path[i - 1][1] - bat, 2)
                    line = f"Step {i:02d}: Move {pos} | Drain -{drain:.1f} | Bat: {bat:.1f}\n"
                self.log_text.insert(tk.END, line)
        else:
            from drone_engine import get_battery_drain
            bat = BATTERY_CAPACITY
            for i, pos in enumerate(path[: self.sim_step + 1]):
                if i == 0:
                    line = f"Step {i:02d}: Takeoff {pos} | Bat: {bat:.1f}/{BATTERY_CAPACITY:.1f}\n"
                else:
                    drain = get_battery_drain(pos)
                    bat = round(bat - drain, 2)
                    flag = " [CRASH!]" if bat < 0 else ""
                    line = f"Step {i:02d}: Move {pos} | Drain -{drain:.1f} | Bat: {bat:.1f}{flag}\n"
                self.log_text.insert(tk.END, line)

        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneDeliveryApp(root)
    root.mainloop()
