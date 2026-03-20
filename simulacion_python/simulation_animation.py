"""
Animación interactiva del seguimiento MPPT P&O.

Slider "Paso" para recorrer la simulación completa de inicio a fin.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

from pv_panel import PanelPV
from boost_converter import BoostConverter
from mppt_p_and_o import MPPT_PandO

# ── Parámetros ────────────────────────────────────────────────────────────────
parametros_panel = {
    'cells_in_series':   36,
    'isc_ref':           5.0,
    'voc_ref':           22.2,
    'vmpp_ref':          18.1,
    'impp_ref':          4.7,
    'ideality_factor':   1.3,
    'series_resistance': 0.221,
}
parametros_boost = {
    'Ci': 22e-6, 'Co': 22e-6, 'L': 330e-6,
    'RL': 60e-3, 'Ron': 35e-3, 'RB': 69e-3, 'dt': 2e-5,
}
parametros_mppt  = {'delta_d': 0.004}

irradiancia   = 1000.0
temperatura   = 25.0
VB            = 24.0
dt            = 2e-5
T_total       = 2.4
num_pasos     = int(T_total / dt)
N_mppt        = 1000
paso_guardado = 50          # guardar 1 de cada 50 → 2 400 frames

# ── Simulación ────────────────────────────────────────────────────────────────
print("Simulando…")
panel = PanelPV(parametros_panel)
boost = BoostConverter(parametros_boost)
mppt  = MPPT_PandO(parametros_mppt)

Vci = 15.0
IL  = panel.calcular(irradiancia, temperatura, Vci)
Vco = VB
D   = 1.0 - 15.0 / VB

t_arr, Vpv_arr, Ipv_arr, Ppv_arr, D_arr = [], [], [], [], []

for paso in range(num_pasos):
    Ipv = panel.calcular(irradiancia, temperatura, Vci)
    Vpv = Vci
    if paso % N_mppt == 0:
        D = mppt.actualizar(Vpv, Ipv, D)
    Vci, IL, Vco = boost.simular(Vci, IL, Vco, VB, Ipv, D)
    if paso % paso_guardado == 0:
        t_arr.append(paso * dt * 1e3)
        Vpv_arr.append(Vpv)
        Ipv_arr.append(Ipv)
        Ppv_arr.append(Vpv * Ipv)
        D_arr.append(D)

t_arr   = np.array(t_arr)
Vpv_arr = np.array(Vpv_arr)
Ipv_arr = np.array(Ipv_arr)
Ppv_arr = np.array(Ppv_arr)
D_arr   = np.array(D_arr)
N_frames = len(t_arr)
print(f"Simulación lista — {N_frames} pasos disponibles.")

# Curva estática del panel
V_curva, I_curva, P_curva = panel.curva_pv(irradiancia, temperatura, n_puntos=400)
idx_mpp  = int(np.argmax(P_curva))
Vmpp_real = V_curva[idx_mpp]
Pmpp_real = P_curva[idx_mpp]
Impp_real = I_curva[idx_mpp]

# ── Figura ────────────────────────────────────────────────────────────────────
BG      = '#0f0f1a'
GRID    = '#2a2a3a'
C_CURVE = '#4fc3f7'
C_MPP   = '#ff6b6b'
C_DOT   = '#ffd54f'
C_VT    = '#69f0ae'
C_PT    = '#ce93d8'
C_IT    = '#ffab91'

fig = plt.figure(figsize=(14, 7.5))
fig.patch.set_facecolor(BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38,
              left=0.06, right=0.97, top=0.91, bottom=0.18)

ax_pv = fig.add_subplot(gs[:, 0])   # Curva P-V (ocupa las 2 filas)
ax_vt = fig.add_subplot(gs[0, 1])   # Vpv(t)
ax_pt = fig.add_subplot(gs[1, 1])   # Ppv(t)
ax_iv = fig.add_subplot(gs[0, 2])   # Curva I-V
ax_dt = fig.add_subplot(gs[1, 2])   # D(t)

def style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, linestyle='--')
    ax.set_title(title, color='white', fontsize=9, pad=4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('#aaaacc')

style(ax_pv, 'Curva P-V')
style(ax_vt, 'Voltaje del panel')
style(ax_pt, 'Potencia del panel')
style(ax_iv, 'Curva I-V')
style(ax_dt, 'Duty Cycle')

fig.suptitle('MPPT P&O — Evolución paso a paso', color='white',
             fontsize=13, fontweight='bold')

# Estáticos
ax_pv.plot(V_curva, P_curva, color=C_CURVE, lw=2)
ax_pv.axvline(Vmpp_real, color=C_MPP, ls=':', lw=1, alpha=0.7)
ax_pv.plot(Vmpp_real, Pmpp_real, 'v', color=C_MPP, ms=9,
           label=f'MPP ({Vmpp_real:.1f}V, {Pmpp_real:.1f}W)')
ax_pv.set_xlabel('Voltaje (V)', color='#aaaacc', fontsize=8)
ax_pv.set_ylabel('Potencia (W)', color='#aaaacc', fontsize=8)
ax_pv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

ax_iv.plot(V_curva, I_curva, color=C_IT, lw=2)
ax_iv.axvline(Vmpp_real, color=C_MPP, ls=':', lw=1, alpha=0.7)
ax_iv.plot(Vmpp_real, Impp_real, 'v', color=C_MPP, ms=9,
           label=f'MPP ({Vmpp_real:.1f}V, {Impp_real:.2f}A)')
ax_iv.set_xlabel('Voltaje (V)', color='#aaaacc', fontsize=8)
ax_iv.set_ylabel('Corriente (A)', color='#aaaacc', fontsize=8)
ax_iv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

ax_vt.axhline(Vmpp_real, color=C_MPP, ls='--', lw=1, alpha=0.8,
              label=f'Vmpp={Vmpp_real:.1f}V')
ax_vt.set_xlim(0, t_arr[-1]); ax_vt.set_ylim(13, 23)
ax_vt.set_xlabel('Tiempo (ms)', color='#aaaacc', fontsize=8)
ax_vt.set_ylabel('Vpv (V)', color='#aaaacc', fontsize=8)
ax_vt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

ax_pt.axhline(Pmpp_real, color=C_MPP, ls='--', lw=1, alpha=0.8,
              label=f'Pmpp={Pmpp_real:.1f}W')
ax_pt.set_xlim(0, t_arr[-1]); ax_pt.set_ylim(0, Pmpp_real * 1.12)
ax_pt.set_xlabel('Tiempo (ms)', color='#aaaacc', fontsize=8)
ax_pt.set_ylabel('Ppv (W)', color='#aaaacc', fontsize=8)
ax_pt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

ax_dt.axhline(1 - Vmpp_real / VB, color=C_MPP, ls='--', lw=1, alpha=0.8,
              label=f'D_ss={1-Vmpp_real/VB:.3f}')
ax_dt.set_xlim(0, t_arr[-1]); ax_dt.set_ylim(0.2, 0.42)
ax_dt.set_xlabel('Tiempo (ms)', color='#aaaacc', fontsize=8)
ax_dt.set_ylabel('D', color='#aaaacc', fontsize=8)
ax_dt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

# Elementos animados
TRAIL = 40
dot_pv,   = ax_pv.plot([], [], 'o',  color=C_DOT, ms=11, zorder=6)
trail_pv, = ax_pv.plot([], [], '-',  color=C_DOT, alpha=0.35, lw=1.4)
dot_iv,   = ax_iv.plot([], [], 'o',  color=C_DOT, ms=9,  zorder=6)
line_vt,  = ax_vt.plot([], [], color=C_VT,  lw=1.8)
dot_vt,   = ax_vt.plot([], [], 'o',  color=C_DOT, ms=7,  zorder=6)
line_pt,  = ax_pt.plot([], [], color=C_PT,  lw=1.8)
dot_pt,   = ax_pt.plot([], [], 'o',  color=C_DOT, ms=7,  zorder=6)
line_dt,  = ax_dt.plot([], [], color='#80cbc4', lw=1.8)
dot_dt,   = ax_dt.plot([], [], 'o',  color=C_DOT, ms=7,  zorder=6)

info = fig.text(0.5, 0.005, '', ha='center', color='#ccccdd', fontsize=9)

def draw(i):
    i = int(np.clip(i, 0, N_frames - 1))
    v, p, cur, d = Vpv_arr[i], Ppv_arr[i], Ipv_arr[i], D_arr[i]
    j0 = max(0, i - TRAIL)

    dot_pv.set_data([v], [p])
    dot_iv.set_data([v], [cur])
    trail_pv.set_data(Vpv_arr[j0:i+1], Ppv_arr[j0:i+1])

    line_vt.set_data(t_arr[:i+1], Vpv_arr[:i+1])
    dot_vt.set_data([t_arr[i]], [v])
    line_pt.set_data(t_arr[:i+1], Ppv_arr[:i+1])
    dot_pt.set_data([t_arr[i]], [p])
    line_dt.set_data(t_arr[:i+1], D_arr[:i+1])
    dot_dt.set_data([t_arr[i]], [d])

    pct = p / Pmpp_real * 100
    info.set_text(
        f'Paso {i}/{N_frames-1}   |   t = {t_arr[i]:.0f} ms   |   '
        f'Vpv = {v:.2f} V   |   Ppv = {p:.1f} W   |   D = {d:.4f}   |   {pct:.1f}% del MPP'
    )
    fig.canvas.draw_idle()

# ── Slider ────────────────────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.12, 0.09, 0.76, 0.025], facecolor='#1a1a2e')
slider = Slider(ax_slider, 'Paso', 0, N_frames - 1,
                valinit=0, valstep=1, color='#4fc3f7')
slider.label.set_color('white')
slider.valtext.set_color(C_DOT)

slider.on_changed(lambda val: draw(int(val)))

# ── Botones Play / Pause ───────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.44, 0.035, 0.06, 0.032], facecolor='#1a1a2e')
ax_pause = fig.add_axes([0.51, 0.035, 0.06, 0.032], facecolor='#1a1a2e')
ax_reset = fig.add_axes([0.37, 0.035, 0.06, 0.032], facecolor='#1a1a2e')

btn_play  = Button(ax_play,  '▶ Play',  color='#1a1a2e', hovercolor='#2a2a4e')
btn_pause = Button(ax_pause, '⏸ Pausa', color='#1a1a2e', hovercolor='#2a2a4e')
btn_reset = Button(ax_reset, '↺ Reset', color='#1a1a2e', hovercolor='#2a2a4e')
for btn in (btn_play, btn_pause, btn_reset):
    btn.label.set_color('white')

state = {'running': False, 'frame': 0}

def step_anim():
    if state['running'] and state['frame'] < N_frames - 1:
        state['frame'] += 1
        slider.set_val(state['frame'])   # dispara draw()

def on_play(event):
    state['running'] = True

def on_pause(event):
    state['running'] = False

def on_reset(event):
    state['running'] = False
    state['frame'] = 0
    slider.set_val(0)

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)

ani = animation.FuncAnimation(
    fig, lambda _: step_anim(),
    interval=20, blit=False, cache_frame_data=False
)

draw(0)
plt.show()
