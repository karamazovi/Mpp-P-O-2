"""
po.py — Perturb & Observe.

Dos modos de operación:
  • Modo D     : step(V, I, D)    → D_nuevo   (sin MPC)
  • Modo Vref  : step_vref(V, I, Vref) → Vref_nuevo  (con MPC activo)

En modo Vref, P&O perturba la referencia de tensión en lugar del duty cycle.
El lazo interno (MPC o steady_state_D) convierte Vref → D.
"""


class PO_MPPT:
    """
    Parámetros
    ----------
    delta_d    : escalón en D      (paper: 0.004 / demo: 0.02)
    delta_vref : escalón en Vref   (usado con MPC, default 0.5 V)
    D_min      : duty mínimo (0.05)
    D_max      : duty máximo (0.95)
    Vmin       : Vref mínimo (0.5 V)
    Vmax       : Vref máximo (22.2 V)
    """

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.delta_d    = p.get('delta_d',    0.004)
        self.delta_vref = p.get('delta_vref', 0.5)
        self.D_min      = p.get('D_min',      0.05)
        self.D_max      = p.get('D_max',      0.95)
        self.Vmin       = p.get('Vmin',       0.5)
        self.Vmax       = p.get('Vmax',       22.2)
        self._P_prev    = None
        self._V_prev    = None
        self._D_last    = p.get('D0', 0.25)
        self._Vref_last = p.get('Vref0', 18.1)

    def reset(self, D0: float = 0.25, Vref0: float = 18.1):
        self._P_prev    = None
        self._V_prev    = None
        self._D_last    = D0
        self._Vref_last = Vref0

    def step(self, V: float, I: float, D: float = None) -> float:
        """Modo D: retorna D actualizado (sin MPC)."""
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

    def step_vref(self, V: float, I: float, Vref: float = None) -> float:
        """Modo Vref: retorna Vref actualizado (para usar con MPC)."""
        if Vref is None:
            Vref = self._Vref_last
        P = V * I

        if self._P_prev is None:
            self._P_prev    = P
            self._V_prev    = V
            self._Vref_last = Vref
            return Vref

        dP   = P    - self._P_prev
        dVref = Vref - self._Vref_last

        if abs(dP) >= 1e-6:
            if (dP > 0 and dVref >= 0) or (dP < 0 and dVref < 0):
                Vref += self.delta_vref
            else:
                Vref -= self.delta_vref

        self._P_prev    = P
        self._V_prev    = V
        Vref = float(max(self.Vmin, min(self.Vmax, Vref)))
        self._Vref_last = Vref
        return Vref
