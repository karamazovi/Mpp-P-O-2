import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.interpolate import interp1d


class PanelPV:
    """
    Modelo de diodo simple (single-diode model) para un panel PV.
    Parámetros calibrados con los datos del artículo:
      Isc=5.0A, Voc=22.2V, Vmpp=18.1V, Impp=4.7A, Ns=36 celdas en serie.

    La calibración resuelve un sistema de 2 ecuaciones (punto MPP + dP/dV=0 en MPP)
    para garantizar que el máximo real de la curva P-V coincida con (Vmpp, Impp).

    La curva I-V se pre-calcula como tabla de lookup interpolada para velocidad.
    """

    def __init__(self, parametros=None):
        if parametros is None:
            parametros = {}
        p = parametros

        self.Ns       = p.get('cells_in_series',   36)
        self.Isc_ref  = p.get('isc_ref',           5.0)
        self.Voc_ref  = p.get('voc_ref',           22.2)
        self.Vmpp_ref = p.get('vmpp_ref',          18.1)
        self.Impp_ref = p.get('impp_ref',          4.7)
        self.n        = p.get('ideality_factor',   1.0)
        self.Rs       = p.get('series_resistance', 0.1)   # valor inicial, se ajusta

        # Constantes físicas
        self.q = 1.602e-19
        self.k = 1.381e-23
        self.T_ref = 298.15  # 25°C en Kelvin
        self.Iph_ref = self.Isc_ref

        # Calibrar todos los parámetros
        self._calibrar()

        # Pre-construir tabla de lookup
        self._lut_G = None
        self._lut_T = None
        self._lut_interp = None
        self._construir_lut(1000.0, 25.0)

    def _Vt(self, T_K):
        return self.n * self.k * T_K / self.q

    def _calibrar(self):
        """
        Calibración de 4 parámetros: Rs, Rp, I0 (con Iph=Isc fijo y n fijo).

        Condiciones:
          1. En Voc (I=0):   Iph = I0*(exp(Voc/NsVt)-1) + Voc/Rp   → I0 analítico dado Rp
          2. En MPP (punto): Impp = Iph - I0*(exp(Vm/NsVt)-1) - Vm/Rp   con Vm=Vmpp+Impp*Rs
          3. En MPP (máximo): dP/dV = 0  →  Impp + Vmpp*dI/dV = 0

        Se resuelve numéricamente para (Rs, Rp).
        """
        Vt   = self._Vt(self.T_ref)
        NsVt = self.Ns * Vt
        Iph  = self.Iph_ref

        def sistema(params):
            Rs, Rp = params
            if Rp < 1.0 or Rs < 0.0:
                return [1e6, 1e6]

            A  = np.exp(self.Voc_ref / NsVt) - 1.0
            I0 = (Iph - self.Voc_ref / Rp) / A
            if I0 <= 0:
                return [1e6, 1e6]

            Vm    = self.Vmpp_ref + self.Impp_ref * Rs
            exp_m = np.exp(Vm / NsVt)

            # Ecuación 1: el punto (Vmpp, Impp) está sobre la curva
            eq1 = Iph - I0 * (exp_m - 1) - Vm / Rp - self.Impp_ref

            # Ecuación 2: dP/dV = 0 en el MPP → Impp + Vmpp*(dI/dV) = 0
            alpha = I0 / NsVt * exp_m
            denom = 1.0 + alpha * Rs + Rs / Rp
            dIdV  = -(alpha + 1.0 / Rp) / denom
            eq2   = self.Impp_ref + self.Vmpp_ref * dIdV

            return [eq1, eq2]

        # Punto de partida: solución analítica sin condición de derivada
        A0  = np.exp(self.Voc_ref / NsVt) - 1.0
        Vm0 = self.Vmpp_ref + self.Impp_ref * self.Rs
        B0  = np.exp(Vm0 / NsVt) - 1.0
        n_Rp = self.Voc_ref * B0 / A0 - Vm0
        d_Rp = self.Impp_ref - Iph * (1.0 - B0 / A0)
        Rp0  = n_Rp / d_Rp if abs(d_Rp) > 1e-12 and n_Rp / d_Rp > 1.0 else 90.0

        sol, info, ier, _ = fsolve(sistema, [self.Rs, Rp0], full_output=True)
        Rs_sol, Rp_sol = sol

        if ier == 1 and Rp_sol > 1.0 and Rs_sol >= 0.0:
            self.Rs = Rs_sol
            self.Rp = Rp_sol
        else:
            # Fallback conservador
            self.Rp = Rp0

        A       = np.exp(self.Voc_ref / NsVt) - 1.0
        self.I0 = (Iph - self.Voc_ref / self.Rp) / A

    # ── Métodos de cálculo ─────────────────────────────────────────────────────

    def _brentq_calcular(self, Iph, Vt, V):
        """Resuelve la ecuación implícita del diodo con brentq (solo para LUT)."""
        def f(I):
            arg = (V + I * self.Rs) / (self.Ns * Vt)
            return Iph - self.I0 * (np.exp(arg) - 1) - (V + I * self.Rs) / self.Rp - I
        try:
            return max(brentq(f, 0.0, Iph * 1.05, xtol=1e-8, maxiter=150), 0.0)
        except ValueError:
            return 0.0

    def _construir_lut(self, G, T, n_puntos=2000):
        """Pre-calcula la curva I-V y construye un interpolador lineal."""
        T_K  = T + 273.15
        Vt   = self._Vt(T_K)
        Iph  = self.Iph_ref * (G / 1000.0)

        V_arr = np.linspace(0.0, self.Voc_ref, n_puntos)
        I_arr = np.array([self._brentq_calcular(Iph, Vt, v) for v in V_arr])

        self._lut_interp = interp1d(V_arr, I_arr, kind='linear',
                                    bounds_error=False, fill_value=0.0)
        self._lut_G = G
        self._lut_T = T

    def calcular(self, irradiancia, temperatura, V):
        """
        Calcula la corriente del panel para un voltaje V dado.
        Usa tabla de lookup interpolada (muy rápido).

        Parámetros:
            irradiancia : W/m²
            temperatura : °C
            V           : voltaje en bornes del panel (V)

        Retorna:
            I : corriente del panel (A)
        """
        if irradiancia != self._lut_G or temperatura != self._lut_T:
            self._construir_lut(irradiancia, temperatura)
        return float(self._lut_interp(np.clip(V, 0.0, self.Voc_ref)))

    def curva_pv(self, irradiancia=1000, temperatura=25, n_puntos=300):
        """
        Genera la curva P-V completa del panel.

        Retorna:
            V_arr, I_arr, P_arr
        """
        if irradiancia != self._lut_G or temperatura != self._lut_T:
            self._construir_lut(irradiancia, temperatura)
        V_arr = np.linspace(0, self.Voc_ref, n_puntos)
        I_arr = np.array([float(self._lut_interp(v)) for v in V_arr])
        P_arr = V_arr * I_arr
        return V_arr, I_arr, P_arr
