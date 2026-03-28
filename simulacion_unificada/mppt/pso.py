"""
pso.py — Particle Swarm Optimization para MPPT.

Evalúa la función de aptitud usando el modelo del panel (si disponible).
Auto-reinicializa el enjambre cuando la irradiancia cambia > 50 W/m².
Compatible con: step(V, I, G, T) → Vref
"""

import numpy as np


class PSO_MPPT:

    def __init__(self, params: dict | None = None, panel=None):
        p = params or {}
        self.n_particles = p.get('n_particles', 10)
        self.n_iter      = p.get('n_iter',      20)
        self.w           = p.get('w',           0.5)
        self.c1          = p.get('c1',          1.5)
        self.c2          = p.get('c2',          1.5)
        self.Vmin        = p.get('Vmin',         0.5)
        self.Vmax        = p.get('Vmax',        22.2)
        self.panel       = panel
        self.G           = p.get('G',        1000.0)
        self.T           = p.get('T',           25.0)
        self._init_swarm()

    def _init_swarm(self):
        n = self.n_particles
        self.pos       = np.linspace(self.Vmin, self.Vmax, n)
        self.vel       = np.zeros(n)
        self.pbest     = self.pos.copy()
        self.pbest_fit = np.full(n, -np.inf)
        self.gbest     = float(np.mean(self.pos))
        self.gbest_fit = -np.inf

    def reset(self, Vref0: float = 18.1):
        self._init_swarm()
        self.gbest = Vref0

    def _fitness(self, V_probe: float, V_meas: float, I_meas: float) -> float:
        if self.panel is not None:
            _, P = self.panel.step(V_probe, self.G, self.T)
            return P
        P_meas = V_meas * I_meas
        return P_meas + I_meas * (V_probe - V_meas)

    def step(self, V: float, I: float,
             G: float = None, T: float = None) -> float:
        # Reinit automático si irradiancia cambia significativamente
        if G is not None and abs(G - self.G) > 50.0:
            self._init_swarm()
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T

        n = self.n_particles
        for _ in range(self.n_iter):
            fit = np.array([self._fitness(v, V, I) for v in self.pos])
            improved = fit > self.pbest_fit
            self.pbest[improved]     = self.pos[improved]
            self.pbest_fit[improved] = fit[improved]
            idx = int(np.argmax(self.pbest_fit))
            if self.pbest_fit[idx] > self.gbest_fit:
                self.gbest_fit = self.pbest_fit[idx]
                self.gbest     = float(self.pbest[idx])
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)
            self.vel = (self.w * self.vel
                        + self.c1 * r1 * (self.pbest - self.pos)
                        + self.c2 * r2 * (self.gbest  - self.pos))
            self.pos = np.clip(self.pos + self.vel, self.Vmin, self.Vmax)

        return float(np.clip(self.gbest, self.Vmin, self.Vmax))
