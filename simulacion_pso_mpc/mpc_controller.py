"""
Controlador MPC (Model Predictive Control) para seguimiento de voltaje.

Usa el modelo de estado estacionario del boost como modelo de predicción:

    Vci_ss(D) = VB · (1 - D)

Función de costo:
    J = (VB·(1-D) - Vref)²  +  lambda_u · (D - D_prev)²

Solución analítica (mínimo global, sin iteraciones):
    D_ss  = 1 - Vref/VB
    D_opt = (VB² · D_ss + lambda_u · D_prev) / (VB² + lambda_u)

El término lambda_u suaviza los cambios de D y controla la velocidad de
convergencia: valores pequeños → convergencia rápida, valores grandes → lenta.

Con lambda_u = 10  y  VB = 24 → converge a D_ss en ~2 llamadas al MPC.
Con lambda_u = 100 y  VB = 24 → converge en ~4 llamadas al MPC.
"""

import numpy as np


class MPC:
    def __init__(self, boost, parametros=None):
        p = parametros or {}
        self.boost    = boost          # se guarda para acceso externo si se requiere
        self.lambda_u = p.get('lambda_u', 10.0)   # peso del término de control
        self.VB_nom   = p.get('VB_nom',  24.0)    # voltaje nominal batería (V)

    def calcular_D(self, Vci, IL, Vco, VB, Ipv, Vref, D_prev):
        """
        Calcula el duty cycle óptimo.

        Parámetros:
            Vci    : voltaje capacitor entrada actual (V)  [no usado — modelo SS]
            IL     : corriente inductor actual (A)         [no usado — modelo SS]
            Vco    : voltaje capacitor salida actual (V)   [no usado — modelo SS]
            VB     : voltaje batería real (V)
            Ipv    : corriente del panel actual (A)        [no usado — modelo SS]
            Vref   : voltaje de referencia del PSO (V)
            D_prev : duty cycle del período anterior

        Retorna:
            D_opt  : duty cycle óptimo (float en [0.05, 0.95])
        """
        # Duty cycle de estado estacionario para alcanzar Vref
        D_ss = 1.0 - Vref / VB

        # Solución analítica del MPC con modelo de estado estacionario
        # J = (VB·(1-D) - Vref)² + lambda_u·(D - D_prev)²
        VB2 = VB ** 2
        D_opt = (VB2 * D_ss + self.lambda_u * D_prev) / (VB2 + self.lambda_u)

        return float(np.clip(D_opt, 0.05, 0.95))
