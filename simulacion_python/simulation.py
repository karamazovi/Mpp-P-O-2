"""
Simulador MPPT con algoritmo Perturbar & Observar (P&O)
para un panel PV con convertidor boost y batería.

Equivalente Python del modelo MATLAB/Simulink del artículo.

Parámetros del artículo:
  Panel  : Vmpp=18.1V, Impp=4.7A, Isc=5.0A, Voc=22.2V, Ns=36
  Boost  : L=330µH, Ci=Co=22µF, RL=60mΩ, Ron=35mΩ, RB=69mΩ
  Batería: VB=24V
  P&O    : delta_d=0.004, D_inicial=0.4
  Sim    : dt=2e-5s, T_total=2.4s, T=25°C
  Irradiancia: perfil con sombreados parciales (igual al Signal 1 de MATLAB)
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from pv_panel import PanelPV
from boost_converter import BoostConverter
from mppt_p_and_o import MPPT_PandO

# ── Parámetros del panel ───────────────────────────────────────────────────────
parametros_panel = {
    'cells_in_series':   36,
    'isc_ref':           5.0,
    'voc_ref':           22.2,
    'vmpp_ref':          18.1,
    'impp_ref':          4.7,
    'ideality_factor':   1.3,
    'series_resistance': 0.221,
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

# ── Parámetros del MPPT P&O ───────────────────────────────────────────────────
parametros_mppt = {
    'delta_d': 0.004,
}

# ── Perfil de irradiancia (sombreado parcial — replica Signal 1 de MATLAB) ─────
def perfil_irradiancia(t):
    """
    Basado en la gráfica MATLAB:
      0.29–0.44s : G baja a 100 W/m² (sombreado)
      0.88–0.99s : G baja a 100 W/m²
      1.18–1.29s : G baja a 100 W/m²
      1.38–1.54s : G baja a 100 W/m²
    Resto       : G = 1000 W/m²
    """
    if (0.29 <= t < 0.44 or
        0.88 <= t < 0.99 or
        1.18 <= t < 1.29 or
        1.38 <= t < 1.54):
        return 100.0
    return 1000.0

# ── Condiciones de simulación ──────────────────────────────────────────────────
temperatura = 25.0     # °C
VB          = 24.0     # Voltaje batería (V)
# D en estado estacionario: D = 1 - Vmpp/VB = 1 - 18.1/24 ≈ 0.246
# Arrancamos un poco por debajo del MPP (Vci=15V) para ver la convergencia
D_init      = 1.0 - 15.0 / VB   # ≈ 0.375
dt          = 2e-5     # Paso de tiempo (s)
T_total     = 2.4      # Tiempo total (s)
num_pasos   = int(T_total / dt)  # 120 000 pasos

# P&O se actualiza cada N pasos.
# τ_L = L/RL = 330µH/60mΩ ≈ 5.5ms → el inductor necesita al menos 3τ ≈ 16ms
# para estabilizarse tras cada perturbación. Usamos 20ms (1000 pasos).
N_mppt = 1000  # actualizar P&O cada 1000 pasos = cada 20ms

# ── Instanciar componentes ─────────────────────────────────────────────────────
panel = PanelPV(parametros_panel)
boost = BoostConverter(parametros_boost)
mppt  = MPPT_PandO(parametros_mppt)

# ── Variables de estado iniciales ─────────────────────────────────────────────
# Inicializar cerca del punto de operación para evitar que el inductor quede a 0
# y el panel suba a Voc sin carga.
Vci = 15.0    # voltaje de arranque (V) — por debajo del MPP (18.1V)
IL  = panel.calcular(1000.0, temperatura, Vci)  # corriente de arranque (A)
Vco = VB      # capacitor de salida ≈ batería (V)
D   = D_init

# ── Buffers de resultados (submuestreo 1:50 para no saturar memoria) ───────────
paso_guardado = 50
t_arr   = []
Vpv_arr = []
Ipv_arr = []
Ppv_arr = []
D_arr   = []
Vco_arr = []
IL_arr  = []
G_arr   = []

print(f"Iniciando simulación: {num_pasos:,} pasos ({T_total}s) ...")

for paso in range(num_pasos):
    t = paso * dt
    irradiancia = perfil_irradiancia(t)

    # 1. Calcular corriente del panel para el voltaje Vci actual
    Ipv = panel.calcular(irradiancia, temperatura, Vci)
    Vpv = Vci   # el voltaje del panel es el voltaje de entrada al boost

    # 2. Actualizar P&O cada N_mppt pasos
    if paso % N_mppt == 0:
        D = mppt.actualizar(Vpv, Ipv, D)

    # 3. Avanzar dinámica del convertidor
    Vci, IL, Vco = boost.simular(Vci, IL, Vco, VB, Ipv, D)

    # 4. Guardar resultados submuestreados
    if paso % paso_guardado == 0:
        t_arr.append(t)
        Vpv_arr.append(Vpv)
        Ipv_arr.append(Ipv)
        Ppv_arr.append(Vpv * Ipv)
        D_arr.append(D)
        Vco_arr.append(Vco)
        IL_arr.append(IL)
        G_arr.append(irradiancia)

print("Simulación completada.")
print(f"  Vpv final : {Vpv_arr[-1]:.3f} V  (referencia Vmpp = {parametros_panel['vmpp_ref']} V)")
print(f"  Ipv final : {Ipv_arr[-1]:.3f} A  (referencia Impp = {parametros_panel['impp_ref']} A)")
print(f"  Ppv final : {Ppv_arr[-1]:.2f} W  (referencia Pmpp = {parametros_panel['vmpp_ref']*parametros_panel['impp_ref']:.2f} W)")
print(f"  D final   : {D_arr[-1]:.4f}")

# ── Guardar resultados en CSV ──────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), 'resultados.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tiempo_s', 'Vpv_V', 'Ipv_A', 'Ppv_W', 'D', 'Vco_V', 'IL_A', 'G_Wm2'])
    for row in zip(t_arr, Vpv_arr, Ipv_arr, Ppv_arr, D_arr, Vco_arr, IL_arr, G_arr):
        writer.writerow([f'{v:.6f}' for v in row])
print(f"Resultados guardados en: {csv_path}")

# ── Gráfica 1: Evolución temporal ─────────────────────────────────────────────
t_ms = np.array(t_arr) * 1e3   # convertir a ms

fig1, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
fig1.suptitle('Evolución temporal – MPPT P&O (panel único)', fontsize=13)

axes[0].plot(t_ms, Vpv_arr, 'b')
axes[0].axhline(parametros_panel['vmpp_ref'], color='r', linestyle='--', label=f"Vmpp={parametros_panel['vmpp_ref']}V")
axes[0].set_ylabel('Vpv (V)')
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].plot(t_ms, Ipv_arr, 'g')
axes[1].axhline(parametros_panel['impp_ref'], color='r', linestyle='--', label=f"Impp={parametros_panel['impp_ref']}A")
axes[1].set_ylabel('Ipv (A)')
axes[1].legend(fontsize=8)
axes[1].grid(True)

axes[2].plot(t_ms, Ppv_arr, 'm')
Pmpp = parametros_panel['vmpp_ref'] * parametros_panel['impp_ref']
axes[2].axhline(Pmpp, color='r', linestyle='--', label=f"Pmpp={Pmpp:.1f}W")
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

# ── Gráfica 2: Curva P-V con trayectoria del P&O ──────────────────────────────
V_curva, _, P_curva = panel.curva_pv(1000.0, temperatura)
idx_mpp = np.argmax(P_curva)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(V_curva, P_curva, 'b-', linewidth=2, label='Curva P-V (G=1000 W/m²)')
ax2.plot(Vpv_arr, Ppv_arr, 'gray', linewidth=0.8, alpha=0.6, label='Trayectoria P&O')
ax2.plot(Vpv_arr[0],  Ppv_arr[0],  'go', markersize=8, label='Inicio')
ax2.plot(Vpv_arr[-1], Ppv_arr[-1], 'rs', markersize=8, label=f'Final ({Vpv_arr[-1]:.2f}V, {Ppv_arr[-1]:.1f}W)')
ax2.plot(V_curva[idx_mpp], P_curva[idx_mpp], 'r^', markersize=10,
         label=f'MPP teórico ({V_curva[idx_mpp]:.2f}V, {P_curva[idx_mpp]:.1f}W)')
ax2.set_xlabel('Voltaje (V)')
ax2.set_ylabel('Potencia (W)')
ax2.set_title('Curva P-V y seguimiento del MPP – P&O')
ax2.legend(fontsize=8)
ax2.grid(True)

plt.tight_layout()
plt.show()
