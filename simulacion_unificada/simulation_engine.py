"""
simulation_engine.py — Motor de simulación unificado.

Orquesta: PanelPV → BoostConverter → MPPT → MPC(opcional)
Soporta generador paso-a-paso iter_steps() para plot en tiempo real.

Períodos de control:
  dt      = 20 µs   (integración boost)
  T_mppt  = 20 ms   (P&O) / 50 ms (INC, PSO)
  T_mpc   =  2 ms   (control D interno, solo si use_mpc=True y algo ≠ po)

Perfiles de irradiancia disponibles:
  'constant'        : G=1000 W/m² constante (demo principal, replica MATLAB scope)
  'partial_shading' : sombreado parcial en t=0.29,0.88,1.18,1.38s
"""

import csv
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pv_panel        import PanelPV
from boost_converter import BoostConverter
from mpc_controller  import MPC
from mppt            import PO_MPPT, INC_MPPT, PSO_MPPT
from mppt.base       import detectar_modo

# ── Perfiles de irradiancia ────────────────────────────────────────────────────
def _perfil_constante(t: float) -> float:
    return 1000.0

def _perfil_sombreado(t: float) -> float:
    if (0.29 <= t < 0.44 or 0.88 <= t < 0.99 or
        1.18 <= t < 1.29 or 1.38 <= t < 1.54):
        return 100.0
    return 1000.0

PERFILES = {
    'constant':        _perfil_constante,
    'partial_shading': _perfil_sombreado,
}

# ── Parámetros por defecto ─────────────────────────────────────────────────────
PANEL_DEFAULT = dict(
    cells_in_series=36, isc_ref=5.0, voc_ref=22.2,
    vmpp_ref=18.1, impp_ref=4.7,
    ideality_factor=1.3, series_resistance=0.221,
)
BOOST_DEFAULT = dict(L=330e-6, Ci=22e-6, Co=22e-6,
                     RL=60e-3, Ron=35e-3, RB=69e-3, RCo=6e-3)


class SimulationEngine:
    """
    Motor de simulación unificado.

    Parámetros
    ----------
    mppt_algo    : 'po' | 'inc' | 'pso'
    use_mpc      : activa MPC interno (ignorado para P&O, que controla D directo)
    irr_profile  : 'constant' | 'partial_shading' | callable f(t)→G
    T_mppt       : período MPPT [s] (None → 20ms para P&O, 50ms para INC/PSO)
    """

    def __init__(self,
                 mppt_algo:    str   = 'po',
                 use_mpc:      bool  = False,
                 panel_params: dict  = None,
                 boost_params: dict  = None,
                 mppt_params:  dict  = None,
                 mpc_params:   dict  = None,
                 T_total:      float = 2.4,
                 dt:           float = 2e-5,
                 T_mpc:        float = 2e-3,
                 T_mppt:       float = None,
                 VB:           float = 24.0,
                 temperatura:  float = 25.0,
                 irr_profile          = 'constant',
                 mppt_obj             = None,
                 mppt_mode:    str   = None):

        self.mppt_algo   = mppt_algo
        # Para modo 'custom', el modo D/Vref se infiere automáticamente
        if mppt_algo == 'custom':
            if mppt_obj is None:
                raise ValueError("mppt_algo='custom' requiere mppt_obj=<instancia>")
            _mode = mppt_mode or detectar_modo(mppt_obj)
            self._custom_obj  = mppt_obj
            self._custom_mode = _mode          # 'po' o 'vref'
        else:
            self._custom_obj  = None
            self._custom_mode = None
        is_d_space = (mppt_algo == 'po') or (mppt_algo == 'custom' and self._custom_mode == 'po')
        self.use_mpc     = use_mpc and not is_d_space
        self.T_total     = T_total
        self.dt          = dt
        self.T_mpc       = T_mpc
        self.VB          = VB
        self.temperatura = temperatura

        # Perfil de irradiancia
        if callable(irr_profile):
            self.irr_fn = irr_profile
        else:
            self.irr_fn = PERFILES.get(irr_profile, _perfil_constante)

        # Período MPPT
        if T_mppt is not None:
            self.T_mppt = T_mppt
        else:
            self.T_mppt = 20e-3 if mppt_algo == 'po' else 50e-3

        # ── Panel ──────────────────────────────────────────────────────────────
        _pp = PANEL_DEFAULT.copy()
        if panel_params:
            _pp.update(panel_params)
        self.panel = PanelPV(_pp)

        # ── Boost ──────────────────────────────────────────────────────────────
        _bp = BOOST_DEFAULT.copy()
        _bp['dt'] = dt
        if boost_params:
            _bp.update(boost_params)
        self.boost = BoostConverter(_bp)

        # ── MPC ────────────────────────────────────────────────────────────────
        self.mpc = MPC(self.boost, mpc_params) if self.use_mpc else None

        # ── MPPT ───────────────────────────────────────────────────────────────
        _mp = mppt_params or {}
        _algo_map = {'po': PO_MPPT, 'inc': INC_MPPT, 'pso': PSO_MPPT}
        if mppt_algo == 'custom':
            self.mppt = self._custom_obj
        elif mppt_algo not in _algo_map:
            raise ValueError(
                f"mppt_algo debe ser 'po', 'inc', 'pso' o 'custom'. "
                f"Para un método propio usa mppt_algo='custom' y mppt_obj=<instancia>.")
        else:
            cls = _algo_map[mppt_algo]
            self.mppt = cls(_mp, panel=self.panel) if mppt_algo == 'pso' else cls(_mp)

        # Potencia de referencia teórica (línea Pref en dashboard)
        Vmpp, Impp, Pmpp = self.panel.mpp(1000.0, temperatura)
        self.Pref  = Pmpp
        self.Vmpp  = Vmpp
        self.Impp  = Impp

        self.results: dict = {}

    # ── Generador paso-a-paso ──────────────────────────────────────────────────
    def iter_steps(self):
        """
        Genera (k, t, Vpv, Ipv, Ppv, D, Vref, Vco) por cada paso.
        Permite actualizar gráficas en tiempo real desde el dashboard.
        """
        num_pasos = int(self.T_total / self.dt)
        paso_mppt = max(1, int(self.T_mppt / self.dt))
        paso_mpc  = max(1, int(self.T_mpc  / self.dt))

        # ── Inicialización (igual al artículo) ─────────────────────────────────
        # Arranca en Vci0=15V (por debajo del MPP) para ver convergencia
        Vci0 = 15.0
        IL0  = self.panel.calcular(1000.0, self.temperatura, Vci0)
        D    = 1.0 - Vci0 / self.VB   # 0.375
        self.boost.reset(Vci0=Vci0, IL0=IL0, Vco0=self.VB)

        # Reset MPPT si tiene método reset
        _is_d_space = (self.mppt_algo == 'po') or \
                      (self.mppt_algo == 'custom' and self._custom_mode == 'po')
        if hasattr(self.mppt, 'reset'):
            if _is_d_space:
                self.mppt.reset(D0=D)
            else:
                self.mppt.reset(Vref0=Vci0)

        Vref = Vci0

        for k in range(num_pasos):
            t    = k * self.dt
            G    = self.irr_fn(t)
            Vpv  = self.boost.Vci
            Ipv, Ppv = self.panel.step(Vpv, G, self.temperatura)

            # ── MPPT ─────────────────────────────────────────────────────────
            if k % paso_mppt == 0:
                if self.mppt_algo == 'po':
                    D    = self.mppt.step(Vpv, Ipv, D)
                    Vref = Vpv
                elif self.mppt_algo == 'inc':
                    Vref = self.mppt.step(Vpv, Ipv)
                elif self.mppt_algo == 'pso':
                    Vref = self.mppt.step(Vpv, Ipv, G=G, T=self.temperatura)
                else:  # custom
                    if self._custom_mode == 'po':
                        D    = self.mppt.step(Vpv, Ipv, D)
                        Vref = Vpv
                    else:
                        # Vref mode: intentar con G,T si los acepta
                        try:
                            Vref = self.mppt.step(Vpv, Ipv, G, self.temperatura)
                        except TypeError:
                            Vref = self.mppt.step(Vpv, Ipv)

            # ── Control interno (Vref → D) ────────────────────────────────────
            if not _is_d_space and k % paso_mpc == 0:
                if self.use_mpc:
                    D = self.mpc.compute(
                        self.boost.Vci, self.boost.IL, self.boost.Vco,
                        self.VB, Ipv, Vref, D
                    )
                else:
                    D = self.boost.steady_state_D(Vref, self.boost.IL, self.VB)

            # ── Boost ─────────────────────────────────────────────────────────
            _Vci, _IL, Vco = self.boost.step(Vpv, Ipv, D, self.VB)

            yield k, t, Vpv, Ipv, Ppv, D, Vref, Vco

    # ── Ejecución completa ─────────────────────────────────────────────────────
    def run(self, verbose: bool = False) -> dict:
        n = int(self.T_total / self.dt)
        t_a   = np.empty(n); Vpv_a = np.empty(n); Ipv_a = np.empty(n)
        Ppv_a = np.empty(n); D_a   = np.empty(n)
        Vref_a= np.empty(n); Vco_a = np.empty(n)

        for k, t, Vpv, Ipv, Ppv, D, Vref, Vco in self.iter_steps():
            t_a[k]=t; Vpv_a[k]=Vpv; Ipv_a[k]=Ipv; Ppv_a[k]=Ppv
            D_a[k]=D; Vref_a[k]=Vref; Vco_a[k]=Vco
            if verbose and k % 20000 == 0:
                print(f"  t={t*1e3:.0f}ms  Vpv={Vpv:.2f}V  Ppv={Ppv:.2f}W  D={D:.4f}")

        self.results = {
            't': t_a, 'Vpv': Vpv_a, 'Ipv': Ipv_a,
            'Ppv': Ppv_a, 'D': D_a, 'Vref': Vref_a, 'Vco': Vco_a,
        }
        return self.results

    # ── Exportar CSV ───────────────────────────────────────────────────────────
    def export_csv(self, filepath: str = 'simulacion.csv'):
        if not self.results:
            raise RuntimeError("Ejecutar run() antes de export_csv()")
        keys = ['t', 'Vpv', 'Ipv', 'Ppv', 'D', 'Vref', 'Vco']
        with open(filepath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(len(self.results['t'])):
                w.writerow([f"{self.results[k][i]:.6f}" for k in keys])
        print(f"[CSV] {filepath}  ({len(self.results['t'])} filas)")

    # ── Comparar vs referencia Simulink ───────────────────────────────────────
    def compare_simulink(self, ref_csv: str, t_start: float = 0.05) -> dict:
        """
        Calcula error de integral de Ppv respecto a CSV de referencia.
        Meta: error < 3%.
        """
        if not self.results:
            raise RuntimeError("Ejecutar run() antes de compare_simulink()")
        if not os.path.exists(ref_csv):
            raise FileNotFoundError(f"No encontrado: {ref_csv}")

        t_ref, P_ref = [], []
        with open(ref_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_ref.append(float(row['t']))
                P_ref.append(float(row.get('Ppv', row.get('power', 0))))
        t_ref = np.array(t_ref)
        P_ref = np.array(P_ref)

        int_ref = np.trapz(P_ref[t_ref >= t_start], t_ref[t_ref >= t_start])
        t_sim   = self.results['t']
        P_sim   = self.results['Ppv']
        int_sim = np.trapz(P_sim[t_sim >= t_start], t_sim[t_sim >= t_start])
        err_pct = abs(int_sim - int_ref) / (abs(int_ref) + 1e-9) * 100.0

        report = {'integral_sim': int_sim, 'integral_ref': int_ref,
                  'error_pct': err_pct, 'passed': err_pct < 3.0}
        print(f"\n{'='*48}")
        print(f"  ∫Ppv simulado  : {int_sim:.4f} W·s")
        print(f"  ∫Ppv referencia: {int_ref:.4f} W·s")
        print(f"  Error integral : {err_pct:.2f}%  "
              f"{'✓ PASS (<3%)' if report['passed'] else '✗ FAIL'}")
        print(f"{'='*48}\n")
        return report
