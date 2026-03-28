"""
boost_converter.py — Convertidor Boost con modelo promediado y pérdidas reales.

Ecuaciones de estado (modelo promediado):
    dVci/dt = (Ipv - IL) / Ci
    dIL/dt  = (Vci - (1-D)*Vco - IL*RL - D*IL*Ron) / L

Vco QUASI-ESTÁTICO: τ = Co*(RB+RCo) ≈ 1.65 µs << dt = 20 µs → Euler inestable.
Se usa el equilibrio instantáneo:
    Vco = VB + (1-D) * IL * (RB + RCo)

Parámetros (artículo):
    L=330µH, Ci=Co=22µF, RL=60mΩ, Ron=35mΩ, RB=69mΩ, RCo=6mΩ, dt=20µs
"""

import numpy as np


class BoostConverter:

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.L    = p.get('L',    330e-6)
        self.Ci   = p.get('Ci',   22e-6)
        self.Co   = p.get('Co',   22e-6)
        self.RL   = p.get('RL',   60e-3)
        self.Ron  = p.get('Ron',  35e-3)
        self.RB   = p.get('RB',   69e-3)
        self.RCo  = p.get('RCo',  6e-3)
        self.dt   = p.get('dt',   2e-5)
        self.VB   = p.get('VB',   24.0)

        # Estados
        self.Vci = p.get('Vci0', 15.0)
        self.IL  = p.get('IL0',   0.0)
        self.Vco = p.get('Vco0', 24.0)

    def reset(self, Vci0: float = 15.0, IL0: float = 0.0, Vco0: float = 24.0):
        self.Vci = Vci0
        self.IL  = IL0
        self.Vco = Vco0

    def step(self, Vpv: float, Ipv: float, D: float, VB: float = None) -> tuple:
        """
        Un paso Euler. Retorna (Vci, IL, Vco) actualizados.
        """
        if VB is None:
            VB = self.VB
        D = float(np.clip(D, 0.0, 1.0))

        # Vco quasi-estático (τ << dt)
        RB_eff   = self.RB + self.RCo
        self.Vco = VB + (1.0 - D) * self.IL * RB_eff

        dIL  = (self.Vci - (1.0 - D) * self.Vco
                - self.IL * self.RL - D * self.IL * self.Ron) / self.L
        dVci = (Ipv - self.IL) / self.Ci

        self.IL  += dIL  * self.dt
        self.Vci += dVci * self.dt
        self.IL   = max(0.0, self.IL)
        self.Vci  = max(0.0, self.Vci)

        return self.Vci, self.IL, self.Vco

    def steady_state_D(self, Vref: float, IL: float, VB: float) -> float:
        """
        D analítico de estado estacionario.
        Resuelve: IL*RB_eff*x² + (VB-IL*Ron)*x + IL*(RL+Ron)-Vref = 0
        donde x = (1-D).
        """
        RB_eff = self.RB + self.RCo
        a_q = IL * RB_eff
        b_q = VB - IL * self.Ron
        c_q = IL * (self.RL + self.Ron) - Vref

        if abs(a_q) > 1e-9:
            disc = max(b_q**2 - 4 * a_q * c_q, 0.0)
            x1 = (-b_q + np.sqrt(disc)) / (2 * a_q)
            x2 = (-b_q - np.sqrt(disc)) / (2 * a_q)
            cands = [1.0 - x for x in (x1, x2) if 0.0 <= x <= 1.0]
            if cands:
                return float(np.clip(min(cands, key=lambda d: abs(d - 0.5)), 0.05, 0.95))

        if abs(VB) > 1e-9:
            return float(np.clip(1.0 - Vref / VB, 0.05, 0.95))
        return 0.25
