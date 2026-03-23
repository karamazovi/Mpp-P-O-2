"""
Controlador MPC (Model Predictive Control) para seguimiento de voltaje.

Usa el modelo de estado estacionario del boost CON compensación resistiva:

    Vci_ss(D) = VB·(1-D) - IL·(RL + D·Ron)

Despejando D (ecuación cuadrática en D):

    D² · IL·Ron  -  D·(VB + IL·RL - IL·Ron)  +  (VB - Vref - IL·RL)  =  0

Se resuelve analíticamente y se elige la raíz en [0, 1].
Sin compensación el MPC apuntaba a D para Vci_ss=Vref ignorando la caída
resistiva IL·(RL+D·Ron), causando Vpv > Vref en ~0.5V.

Función de costo con suavizado:
    J = (Vci_ss(D) - Vref)²  +  lambda_u · (D - D_prev)²
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
        Calcula el duty cycle óptimo con compensación de caída resistiva.

        Parámetros:
            Vci    : voltaje capacitor entrada actual (V)
            IL     : corriente inductor actual (A)  ← usado para compensación
            Vco    : voltaje capacitor salida actual (V)
            VB     : voltaje batería real (V)
            Ipv    : corriente del panel actual (A)
            Vref   : voltaje de referencia del PSO (V)
            D_prev : duty cycle del período anterior

        Retorna:
            D_opt  : duty cycle óptimo (float en [0.05, 0.95])
        """
        RL  = self.boost.RL
        Ron = self.boost.Ron
        RB  = self.boost.RB

        # Modelo SS completo (de dIL/dt=0 con Vco quasi-estático):
        #   Vci_ss = (1-D)*VB + IL*[(1-D)²*RB + RL + D*Ron]
        #
        # Con x = (1-D):
        #   Vref = x*VB + IL*(x²*RB + RL + (1-x)*Ron)
        #   IL*RB*x² + (VB - IL*Ron)*x + IL*(RL+Ron) - Vref = 0
        a_q = IL * RB
        b_q = VB - IL * Ron
        c_q = IL * (RL + Ron) - Vref

        if abs(a_q) > 1e-9:
            disc = b_q**2 - 4*a_q*c_q
            if disc >= 0:
                x1 = (-b_q + np.sqrt(disc)) / (2*a_q)
                x2 = (-b_q - np.sqrt(disc)) / (2*a_q)
                # x = (1-D) debe estar en [0.05, 0.95]
                r1 = float(np.clip(1.0 - x1, 0.05, 0.95))
                r2 = float(np.clip(1.0 - x2, 0.05, 0.95))
                D_ss = r1 if abs(r1 - D_prev) < abs(r2 - D_prev) else r2
            else:
                D_ss = float(np.clip(1.0 - (-b_q / (2*a_q)), 0.05, 0.95))
        else:
            # IL ≈ 0: sin pérdidas resistivas
            D_ss = 1.0 - Vref / VB

        # Suavizado con lambda_u (evita saltos bruscos de D)
        VB2   = VB ** 2
        D_opt = (VB2 * D_ss + self.lambda_u * D_prev) / (VB2 + self.lambda_u)

        return float(np.clip(D_opt, 0.05, 0.95))
