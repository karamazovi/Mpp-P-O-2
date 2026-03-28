"""
inc.py — Conductancia Incremental (INC).

Condición de MPP: dI/dV + I/V = 0
Opera en espacio Vref. Compatible con: step(V, I) → Vref
"""


class INC_MPPT:

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.Vmin  = p.get('Vmin',  0.5)
        self.Vmax  = p.get('Vmax',  22.2)
        self.delta = p.get('delta', 0.5)
        self.Vref  = p.get('Vref0', 18.1)
        self.eps   = p.get('eps',   1e-4)
        self._V_prev = None
        self._I_prev = None

    def reset(self, Vref0: float = 18.1):
        self.Vref    = Vref0
        self._V_prev = None
        self._I_prev = None

    def step(self, V: float, I: float) -> float:
        if self._V_prev is None:
            self._V_prev = V
            self._I_prev = I
            return self.Vref

        dV = V - self._V_prev
        dI = I - self._I_prev

        if abs(dV) < self.eps:
            if abs(dI) >= self.eps:
                self.Vref += self.delta if dI > 0 else -self.delta
        else:
            inc  = dI / dV
            inst = -I / V if abs(V) > self.eps else 0.0
            if abs(inc - inst) >= self.eps:
                self.Vref += self.delta if inc > inst else -self.delta

        self._V_prev = V
        self._I_prev = I
        self.Vref = float(max(self.Vmin, min(self.Vmax, self.Vref)))
        return self.Vref
