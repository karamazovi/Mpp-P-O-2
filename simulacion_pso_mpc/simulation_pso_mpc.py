"""
Simulador MPPT con PSO + MPC para un panel PV con convertidor boost y batería.

Arquitectura de control de dos niveles:
  Nivel externo (PSO) : encuentra Vref = voltaje MPP cada T_pso = 50ms
  Nivel interno (MPC) : lleva Vpv → Vref usando el modelo del boost cada T_mpc = 2ms

Mismo perfil de irradiancia que simulation.py para comparación directa con P&O.
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Importar módulos del panel y boost desde la carpeta hermana
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulacion_python'))
from pv_panel import PanelPV
from boost_converter import BoostConverter

from pso_mppt import PSO_MPPT
from mpc_controller import MPC

# ── Perfil de irradiancia (idéntico a simulation.py) ──────────────────────────
def perfil_irradiancia(t):
    if (0.29 <= t < 0.44 or
        0.88 <= t < 0.99 or
        1.18 <= t < 1.29 or
        1.38 <= t < 1.54):
        return 100.0
    return 1000.0

# ── Parámetros del panel ───────────────────────────────────────────────────────
parametros_panel = {
    'cells_in_series':   36,
    'isc_ref':           5.0,
    'voc_ref':           22.2,
    'vmpp_ref':          18.1,
    'impp_ref':          4.7,
    'ideality_factor':   1.0,
    'series_resistance': 0.1,
}

# ── Parámetros del convertidor boost ──────────────────────────────────────────
parametros_boost = {
    'Ci':  22e-6,
    'Co':  22e-6,
    'L':   330e-6,
    'RL':  60e-3,
    'Ron': 35e-3,
    'RB':  69e-3,
    'dt':  2e-5,
}

# ── Parámetros PSO ─────────────────────────────────────────────────────────────
parametros_pso = {
    'n_particles': 10,
    'w':           0.5,
    'c1':          1.5,
    'c2':          1.5,
    'n_iter':      20,
    'Vmin':        0.5,
    'Vmax':        22.2,
}

# ── Parámetros MPC ─────────────────────────────────────────────────────────────
parametros_mpc = {
    'lambda_u': 10.0,   # peso del término de control (10 → converge en ~2 llamadas)
    'VB_nom':   24.0,
}

# ── Condiciones de simulación ──────────────────────────────────────────────────
temperatura  = 25.0
VB           = 24.0
dt           = 2e-5
T_total      = 2.4
num_pasos    = int(T_total / dt)   # 120 000

# PSO busca nuevo Vref cada 50ms (2500 pasos)
N_pso  = 2500

# MPC calcula nuevo D cada 2ms (100 pasos)
N_mpc  = 100

# ── Instanciar componentes ─────────────────────────────────────────────────────
panel = PanelPV(parametros_panel)
boost = BoostConverter(parametros_boost)
pso   = PSO_MPPT(parametros_pso)
mpc   = MPC(boost, parametros_mpc)

# ── Estado inicial ─────────────────────────────────────────────────────────────
Vci  = 15.0
IL   = panel.calcular(1000.0, temperatura, Vci)
Vco  = VB
D    = 1.0 - 15.0 / VB   # ≈ 0.375
Vref = 15.0               # referencia inicial del PSO

# ── Buffers de resultados ──────────────────────────────────────────────────────
paso_guardado = 50
t_arr    = []
Vpv_arr  = []
Ipv_arr  = []
Ppv_arr  = []
D_arr    = []
Vref_arr = []
G_arr    = []

print(f"Iniciando simulación PSO+MPC: {num_pasos:,} pasos ({T_total}s) ...")

G_prev = 1000.0

for paso in range(num_pasos):
    t = paso * dt
    irradiancia = perfil_irradiancia(t)

    # Detectar cambio brusco de irradiancia → reiniciar PSO
    if irradiancia != G_prev:
        pso.reiniciar()
        G_prev = irradiancia

    # 1. PSO: buscar nuevo Vref cada N_pso pasos
    if paso % N_pso == 0:
        Vref = pso.buscar_mpp(panel, irradiancia, temperatura)

    # 2. Calcular corriente del panel
    Ipv = panel.calcular(irradiancia, temperatura, Vci)
    Vpv = Vci

    # 3. MPC: calcular D óptimo cada N_mpc pasos
    if paso % N_mpc == 0:
        D = mpc.calcular_D(Vci, IL, Vco, VB, Ipv, Vref, D)

    # 4. Avanzar dinámica del convertidor
    Vci, IL, Vco = boost.simular(Vci, IL, Vco, VB, Ipv, D)

    # 5. Guardar resultados submuestreados
    if paso % paso_guardado == 0:
        t_arr.append(t)
        Vpv_arr.append(Vpv)
        Ipv_arr.append(Ipv)
        Ppv_arr.append(Vpv * Ipv)
        D_arr.append(D)
        Vref_arr.append(Vref)
        G_arr.append(irradiancia)

print("Simulación completada.")
print(f"  Vpv final : {Vpv_arr[-1]:.3f} V  (referencia Vmpp = {parametros_panel['vmpp_ref']} V)")
print(f"  Ipv final : {Ipv_arr[-1]:.3f} A  (referencia Impp = {parametros_panel['impp_ref']} A)")
print(f"  Ppv final : {Ppv_arr[-1]:.2f} W")
print(f"  Vref PSO  : {Vref_arr[-1]:.3f} V")
print(f"  D final   : {D_arr[-1]:.4f}")

# ── Guardar CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), 'resultados_pso_mpc.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tiempo_s', 'Vpv_V', 'Ipv_A', 'Ppv_W', 'D', 'Vref_V', 'G_Wm2'])
    for row in zip(t_arr, Vpv_arr, Ipv_arr, Ppv_arr, D_arr, Vref_arr, G_arr):
        writer.writerow([f'{v:.6f}' for v in row])
print(f"Resultados guardados en: {csv_path}")

# ── Gráfica 1: Evolución temporal ─────────────────────────────────────────────
t_ms = np.array(t_arr) * 1e3

fig1, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
fig1.suptitle('Evolución temporal – MPPT PSO+MPC (panel único)', fontsize=13)

axes[0].plot(t_ms, Vpv_arr, 'b', label='Vpv')
axes[0].plot(t_ms, Vref_arr, 'r--', lw=1, label='Vref PSO')
axes[0].axhline(parametros_panel['vmpp_ref'], color='gray', ls=':', lw=1,
                label=f"Vmpp={parametros_panel['vmpp_ref']}V")
axes[0].set_ylabel('Vpv (V)')
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].plot(t_ms, Ipv_arr, 'g')
axes[1].axhline(parametros_panel['impp_ref'], color='r', ls='--',
                label=f"Impp={parametros_panel['impp_ref']}A")
axes[1].set_ylabel('Ipv (A)')
axes[1].legend(fontsize=8)
axes[1].grid(True)

Pmpp = parametros_panel['vmpp_ref'] * parametros_panel['impp_ref']
axes[2].plot(t_ms, Ppv_arr, 'm')
axes[2].axhline(Pmpp, color='r', ls='--', label=f"Pmpp={Pmpp:.1f}W")
axes[2].set_ylabel('Ppv (W)')
axes[2].legend(fontsize=8)
axes[2].grid(True)

axes[3].plot(t_ms, D_arr, 'k')
axes[3].set_ylabel('Duty Cycle D')
axes[3].grid(True)

axes[4].plot(t_ms, G_arr, color='orange')
axes[4].set_ylabel('G (W/m²)')
axes[4].set_ylim(-50, 1100)
axes[4].set_xlabel('Tiempo (ms)')
axes[4].grid(True)

plt.tight_layout()

# ── Gráfica 2: Curva P-V con trayectoria ──────────────────────────────────────
V_curva, _, P_curva = panel.curva_pv(1000.0, temperatura)
idx_mpp = np.argmax(P_curva)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(V_curva, P_curva, 'b-', lw=2, label='Curva P-V (G=1000 W/m²)')
ax2.plot(Vpv_arr, Ppv_arr, 'gray', lw=0.8, alpha=0.6, label='Trayectoria PSO+MPC')
ax2.plot(Vpv_arr[0],  Ppv_arr[0],  'go', ms=8, label='Inicio')
ax2.plot(Vpv_arr[-1], Ppv_arr[-1], 'rs', ms=8,
         label=f'Final ({Vpv_arr[-1]:.2f}V, {Ppv_arr[-1]:.1f}W)')
ax2.plot(V_curva[idx_mpp], P_curva[idx_mpp], 'r^', ms=10,
         label=f'MPP teórico ({V_curva[idx_mpp]:.2f}V, {P_curva[idx_mpp]:.1f}W)')
ax2.set_xlabel('Voltaje (V)')
ax2.set_ylabel('Potencia (W)')
ax2.set_title('Curva P-V y seguimiento del MPP – PSO+MPC')
ax2.legend(fontsize=8)
ax2.grid(True)

plt.tight_layout()
plt.show()
