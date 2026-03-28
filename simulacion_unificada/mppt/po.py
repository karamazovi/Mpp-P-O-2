"""
po.py — Perturb & Observe en espacio D (duty cycle).

Perturba D con delta_d y observa si la potencia sube o baja.
Compatible con: step(V, I, D) → D_nuevo
"""


class PO_MPPT:
    """
    Parámetros
    ----------
    delta_d : escalón en D (paper: 0.004 / demo visible: 0.02)
    D_min   : duty mínimo (0.05)
    D_max   : duty máximo (0.95)
    """

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.delta_d = p.get('delta_d', 0.004)
        self.D_min   = p.get('D_min',   0.05)
        self.D_max   = p.get('D_max',   0.95)
        self._P_prev = None
        self._V_prev = None
        self._D_last = p.get('D0', 0.25)

    def reset(self, D0: float = 0.25):
        self._P_prev = None
        self._V_prev = None
        self._D_last = D0

    def step(self, V: float, I: float, D: float = None) -> float:
        """Retorna D actualizado."""
        if D is None:
            D = self._D_last
        P = V * I

        if self._P_prev is None:
            self._P_prev = P
            self._V_prev = V
            self._D_last = D
            return D

        dP = P - self._P_prev
        dV = V - self._V_prev

        if abs(dP) >= 1e-6:
            if dP > 0:
                D += self.delta_d if dV < 0 else -self.delta_d
            else:
                D += self.delta_d if dV > 0 else -self.delta_d

        self._P_prev = P
        self._V_prev = V
        D = float(max(self.D_min, min(self.D_max, D)))
        self._D_last = D
        return D
