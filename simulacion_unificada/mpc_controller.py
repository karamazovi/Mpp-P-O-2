"""
mpc_controller.py — MPC analítico para seguimiento de voltaje del panel PV.

Calcula D de estado estacionario resolviendo la ecuación cuadrática exacta:
    IL*RB*x² + (VB - IL*Ron)*x + IL*(RL+Ron) - Vref = 0   con x = (1-D)

Suavizado: D_opt = (VB² * D_ss + λu * D_prev) / (VB² + λu)

Complejidad O(1) por llamada — sin búsqueda en grid.
"""

import numpy as np


class MPC:

    def __init__(self, boost, parametros=None):
        p = parametros or {}
        self.boost    = boost
        self.lambda_u = p.get('lambda_u', 10.0)
        self.VB_nom   = p.get('VB_nom',   24.0)

    def calcular_D(self, Vci, IL, Vco, VB, Ipv, Vref, D_prev):
        RL  = self.boost.RL
        Ron = self.boost.Ron
        RB  = self.boost.RB + self.boost.RCo   # incluye ESR

        a_q = IL * RB
        b_q = VB - IL * Ron
        c_q = IL * (RL + Ron) - Vref

        if abs(a_q) > 1e-9:
            disc = b_q**2 - 4 * a_q * c_q
            if disc >= 0:
                x1 = (-b_q + np.sqrt(disc)) / (2 * a_q)
                x2 = (-b_q - np.sqrt(disc)) / (2 * a_q)
                r1 = float(np.clip(1.0 - x1, 0.05, 0.95))
                r2 = float(np.clip(1.0 - x2, 0.05, 0.95))
                D_ss = r1 if abs(r1 - D_prev) < abs(r2 - D_prev) else r2
            else:
                D_ss = float(np.clip(1.0 - (-b_q / (2 * a_q)), 0.05, 0.95))
        else:
            D_ss = float(np.clip(1.0 - Vref / VB, 0.05, 0.95))

        VB2   = VB ** 2
        D_opt = (VB2 * D_ss + self.lambda_u * D_prev) / (VB2 + self.lambda_u)
        return float(np.clip(D_opt, 0.05, 0.95))

    # alias para compatibilidad con SimulationEngine
    def compute(self, Vci, IL, Vco, VB, Ipv, Vref, D_prev):
        return self.calcular_D(Vci, IL, Vco, VB, Ipv, Vref, D_prev)
