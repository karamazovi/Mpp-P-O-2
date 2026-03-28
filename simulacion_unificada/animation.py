"""
animation.py — Animación TIEMPO REAL del seguimiento MPPT.

La simulación corre en un hilo worker mientras FuncAnimation actualiza
las gráficas cada 30ms (igual que el MATLAB Scope).

Al terminar la simulación:
  - Render final de alta resolución (todos los puntos)
  - Slider activo para scrubbing paso a paso
  - Botones Play / Pausa / Reset activos

Layout (GridSpec 3×3):
  Izquierda  : Curva P-V estática + punto móvil + trail
  Centro     : Vpv(t), Ppv(t), G(t)
  Derecha    : Curva I-V + punto móvil, Ipv(t), D(t)

Uso standalone:
  python animation.py                          # P&O, G=1000 constante
  python animation.py --algo inc               # INC
  python animation.py --algo pso --irr partial_shading

Uso desde dashboard: botón "Animación paso a paso" llama a launch().
"""

import os
import sys
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as _mplanim
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from simulation_engine import SimulationEngine

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = '#0f0f1a'
GRID    = '#2a2a3a'
C_CURVE = '#4fc3f7'
C_MPP   = '#ff6b6b'
C_DOT   = '#ffd54f'
C_VT    = '#69f0ae'
C_PT    = '#ce93d8'
C_IT    = '#ffab91'
C_DT    = '#80cbc4'
C_GT    = '#ffcc80'

TRAIL      = 80     # puntos de estela sobre la curva P-V durante sim en vivo
RT_DISPLAY = 1000   # puntos máx en las líneas de tiempo durante la sim
FINAL_PTS  = 6000   # puntos para el render final (después de sim)


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_title(title, color='white', fontsize=9, pad=4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('#aaaacc')


# ── Función principal ─────────────────────────────────────────────────────────
def launch(engine: SimulationEngine = None,
           algo: str = 'po',
           irr: str = 'constant'):
    """
    Lanza la ventana de animación en tiempo real.

    Parámetros
    ----------
    engine : SimulationEngine ya construido (NO hace falta haberlo ejecutado).
             Si None, se construye internamente con algo e irr.
    algo   : 'po' | 'inc' | 'pso'  (solo si engine=None)
    irr    : 'constant' | 'partial_shading'
    """
    # ── Construir engine si no viene dado ─────────────────────────────────────
    if engine is None:
        panel_p = dict(cells_in_series=36, isc_ref=5.0, voc_ref=22.2,
                       vmpp_ref=18.1, impp_ref=4.7,
                       ideality_factor=1.3, series_resistance=0.221)
        boost_p = dict(L=330e-6, Ci=22e-6, Co=22e-6,
                       RL=60e-3, Ron=35e-3, RB=69e-3, RCo=6e-3)
        mppt_p  = {'po': {'delta_d': 0.02},
                   'inc': {'delta': 0.5},
                   'pso': {'n_particles': 10, 'n_iter': 20}}.get(algo, {})
        engine  = SimulationEngine(
            mppt_algo=algo, panel_params=panel_p, boost_params=boost_p,
            mppt_params=mppt_p, T_total=2.4, irr_profile=irr)

    panel  = engine.panel
    Vmpp   = engine.Vmpp
    Impp   = engine.Impp
    Pmpp   = engine.Pref
    VB     = engine.VB
    irr_fn = engine.irr_fn
    n_total = int(engine.T_total / engine.dt)  # 120 000 pasos
    T_ms   = engine.T_total * 1e3

    V_curva, I_curva, P_curva = panel.curva_pv(1000.0, 25.0, n_puntos=400)

    # ── Estado compartido worker ↔ FuncAnimation ──────────────────────────────
    _raw  = []          # list of (t_ms, Vpv, Ipv, Ppv, D, Vref, Vco)
    _state = {'done': False, 'finalized': False, 'scrub': False, 'scrub_i': 0}

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 8.5))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.58, wspace=0.38,
                   left=0.06, right=0.97, top=0.91, bottom=0.19)

    ax_pv = fig.add_subplot(gs[:, 0])
    ax_vt = fig.add_subplot(gs[0, 1])
    ax_pt = fig.add_subplot(gs[1, 1])
    ax_gt = fig.add_subplot(gs[2, 1])
    ax_iv = fig.add_subplot(gs[0, 2])
    ax_it = fig.add_subplot(gs[1, 2])
    ax_dt = fig.add_subplot(gs[2, 2])

    for ax, title in [(ax_pv, 'Curva P-V'),
                      (ax_vt, 'Voltaje del panel  Vpv'),
                      (ax_pt, 'Potencia del panel  Ppv'),
                      (ax_gt, 'Irradiancia  G'),
                      (ax_iv, 'Curva I-V'),
                      (ax_it, 'Corriente  Ipv'),
                      (ax_dt, 'Duty Cycle  D')]:
        _style(ax, title)

    fig.suptitle(
        f'MPPT {algo.upper()} — tiempo real  ({irr})',
        color='white', fontsize=12, fontweight='bold')

    # Curvas estáticas P-V e I-V
    ax_pv.plot(V_curva, P_curva, color=C_CURVE, lw=2)
    ax_pv.axvline(Vmpp, color=C_MPP, ls=':', lw=1, alpha=0.7)
    ax_pv.plot(Vmpp, Pmpp, 'v', color=C_MPP, ms=9,
               label=f'MPP ({Vmpp:.1f} V, {Pmpp:.1f} W)')
    ax_pv.set_xlabel('Voltaje (V)', color='#aaaacc', fontsize=8)
    ax_pv.set_ylabel('Potencia (W)', color='#aaaacc', fontsize=8)
    ax_pv.set_xlim(0, V_curva[-1] * 1.05)
    ax_pv.set_ylim(0, P_curva.max() * 1.15)
    ax_pv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    ax_iv.plot(V_curva, I_curva, color=C_IT, lw=2)
    ax_iv.axvline(Vmpp, color=C_MPP, ls=':', lw=1, alpha=0.7)
    ax_iv.plot(Vmpp, Impp, 'v', color=C_MPP, ms=9,
               label=f'MPP ({Vmpp:.1f} V, {Impp:.2f} A)')
    ax_iv.set_xlabel('Voltaje (V)', color='#aaaacc', fontsize=8)
    ax_iv.set_ylabel('Corriente (A)', color='#aaaacc', fontsize=8)
    ax_iv.set_xlim(0, V_curva[-1] * 1.05)
    ax_iv.set_ylim(0, I_curva[0] * 1.15)
    ax_iv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    # Líneas de referencia en ejes de tiempo
    ax_vt.axhline(Vmpp, color=C_MPP, ls='--', lw=1, alpha=0.8,
                  label=f'Vmpp = {Vmpp:.1f} V')
    ax_vt.set_xlim(0, T_ms); ax_vt.set_ylim(10, 24)
    ax_vt.set_xlabel('t (ms)', color='#aaaacc', fontsize=8)
    ax_vt.set_ylabel('V', color='#aaaacc', fontsize=8)
    ax_vt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    ax_pt.axhline(Pmpp, color=C_MPP, ls='--', lw=1, alpha=0.8,
                  label=f'Pref = {Pmpp:.1f} W')
    ax_pt.set_xlim(0, T_ms); ax_pt.set_ylim(0, Pmpp * 1.15)
    ax_pt.set_xlabel('t (ms)', color='#aaaacc', fontsize=8)
    ax_pt.set_ylabel('W', color='#aaaacc', fontsize=8)
    ax_pt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    ax_it.axhline(Impp, color=C_MPP, ls='--', lw=1, alpha=0.8,
                  label=f'Impp = {Impp:.2f} A')
    ax_it.set_xlim(0, T_ms); ax_it.set_ylim(0, Impp * 1.3)
    ax_it.set_xlabel('t (ms)', color='#aaaacc', fontsize=8)
    ax_it.set_ylabel('A', color='#aaaacc', fontsize=8)
    ax_it.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    D_ss = 1.0 - Vmpp / VB
    ax_dt.axhline(D_ss, color=C_MPP, ls='--', lw=1, alpha=0.8,
                  label=f'D_ss = {D_ss:.3f}')
    ax_dt.set_xlim(0, T_ms); ax_dt.set_ylim(0.1, 0.95)
    ax_dt.set_xlabel('t (ms)', color='#aaaacc', fontsize=8)
    ax_dt.set_ylabel('D', color='#aaaacc', fontsize=8)
    ax_dt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    ax_gt.axhline(1000, color=C_MPP, ls='--', lw=1, alpha=0.6,
                  label='G = 1000 W/m²')
    ax_gt.set_xlim(0, T_ms); ax_gt.set_ylim(-50, 1150)
    ax_gt.set_xlabel('t (ms)', color='#aaaacc', fontsize=8)
    ax_gt.set_ylabel('W/m²', color='#aaaacc', fontsize=8)
    ax_gt.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

    # ── Artistas animados ─────────────────────────────────────────────────────
    dot_pv,   = ax_pv.plot([], [], 'o',  color=C_DOT, ms=11, zorder=6)
    trail_pv, = ax_pv.plot([], [], '-',  color=C_DOT, alpha=0.4, lw=1.5)
    dot_iv,   = ax_iv.plot([], [], 'o',  color=C_DOT, ms=9,  zorder=6)
    line_vt,  = ax_vt.plot([], [], color=C_VT,  lw=1.2)
    dot_vt,   = ax_vt.plot([], [], 'o',  color=C_DOT, ms=7)
    line_pt,  = ax_pt.plot([], [], color=C_PT,  lw=1.2)
    dot_pt,   = ax_pt.plot([], [], 'o',  color=C_DOT, ms=7)
    line_it,  = ax_it.plot([], [], color=C_IT,  lw=1.2)
    dot_it,   = ax_it.plot([], [], 'o',  color=C_DOT, ms=7)
    line_dt,  = ax_dt.plot([], [], color=C_DT,  lw=1.2)
    dot_dt,   = ax_dt.plot([], [], 'o',  color=C_DOT, ms=7)
    line_gt,  = ax_gt.plot([], [], color=C_GT,  lw=1.2)
    dot_gt,   = ax_gt.plot([], [], 'o',  color=C_DOT, ms=7)

    info = fig.text(0.5, 0.005, 'Iniciando simulación…',
                    ha='center', color='#ccccdd', fontsize=9)

    # ── Render final (alta resolución, activado al terminar la sim) ───────────
    def _final_render():
        n    = len(_raw)
        step = max(1, n // FINAL_PTS)
        hist = _raw[::step]
        arr  = np.array(hist)
        t_h  = arr[:, 0];  V_h = arr[:, 1];  I_h = arr[:, 2]
        P_h  = arr[:, 3];  D_h = arr[:, 4]
        G_h  = np.array([irr_fn(t / 1e3) for t in t_h])

        # Punto final = último dato
        last = _raw[-1]
        t_l, Vl, Il, Pl, Dl = last[0], last[1], last[2], last[3], last[4]

        dot_pv.set_data([Vl], [Pl])
        trail_pv.set_data(V_h[-TRAIL:], P_h[-TRAIL:])
        dot_iv.set_data([Vl], [Il])
        line_vt.set_data(t_h, V_h);   dot_vt.set_data([t_l], [Vl])
        line_pt.set_data(t_h, P_h);   dot_pt.set_data([t_l], [Pl])
        line_it.set_data(t_h, I_h);   dot_it.set_data([t_l], [Il])
        line_dt.set_data(t_h, D_h);   dot_dt.set_data([t_l], [Dl])
        line_gt.set_data(t_h, G_h);   dot_gt.set_data([t_l], [irr_fn(t_l / 1e3)])

        pct = Pl / Pmpp * 100 if Pmpp > 0 else 0
        N   = len(_raw)
        info.set_text(
            f'Completado  {N:,} muestras  |  Vpv = {Vl:.2f} V  |  '
            f'Ppv = {Pl:.1f} W  |  D = {Dl:.4f}  |  {pct:.1f}% MPP  '
            f'← Usa el slider para analizar')

        # Actualizar límites con datos reales
        ax_vt.set_ylim(max(0, V_h.min() - 1), V_h.max() + 1)
        ax_pt.set_ylim(0, max(Pmpp * 1.15, P_h.max() * 1.05))
        ax_it.set_ylim(0, I_h.max() * 1.15)
        ax_dt.set_ylim(max(0, D_h.min() - 0.03), D_h.max() + 0.03)

        # Activar slider y botones
        slider.valmax = N - 1
        slider.ax.set_xlim(0, N - 1)
        slider.set_val(N - 1)
        slider.set_active(True)
        for btn in (btn_play, btn_pau, btn_rst):
            btn.set_active(True)
            btn.ax.set_facecolor('#1a1a2e')

        fig.canvas.draw()

    # ── draw_scrub: navegar con slider (solo post-sim) ────────────────────────
    def draw_scrub(i):
        if not _state['done']:
            return
        n   = len(_raw)
        i   = int(np.clip(i, 0, n - 1))
        j0  = max(0, i - TRAIL)
        row = _raw[i]
        t_l, Vl, Il, Pl, Dl = row[0], row[1], row[2], row[3], row[4]
        Gl  = irr_fn(t_l / 1e3)

        trail_v = [_raw[k][1] for k in range(j0, i + 1)]
        trail_p = [_raw[k][3] for k in range(j0, i + 1)]

        # Historia hasta el frame i (submuestreada)
        step = max(1, i // RT_DISPLAY) if i > 0 else 1
        hist = _raw[:i+1:step]
        arr  = np.array(hist) if hist else np.zeros((1, 7))
        t_h  = arr[:, 0];  V_h = arr[:, 1];  I_h = arr[:, 2]
        P_h  = arr[:, 3];  D_h = arr[:, 4]
        G_h  = np.array([irr_fn(t / 1e3) for t in t_h])

        dot_pv.set_data([Vl], [Pl])
        trail_pv.set_data(trail_v, trail_p)
        dot_iv.set_data([Vl], [Il])
        line_vt.set_data(t_h, V_h);   dot_vt.set_data([t_l], [Vl])
        line_pt.set_data(t_h, P_h);   dot_pt.set_data([t_l], [Pl])
        line_it.set_data(t_h, I_h);   dot_it.set_data([t_l], [Il])
        line_dt.set_data(t_h, D_h);   dot_dt.set_data([t_l], [Dl])
        line_gt.set_data(t_h, G_h);   dot_gt.set_data([t_l], [Gl])

        pct = Pl / Pmpp * 100 if Pmpp > 0 else 0
        info.set_text(
            f'Frame {i}/{len(_raw)-1}  |  t = {t_l:.0f} ms  |  '
            f'Vpv = {Vl:.2f} V  |  Ppv = {Pl:.1f} W  |  '
            f'D = {Dl:.4f}  |  {pct:.1f}% MPP')
        fig.canvas.draw_idle()

    # ── FuncAnimation update (tiempo real) ────────────────────────────────────
    def _update(_frame):
        # Si estamos en modo scrubbing, no hacer nada (slider controla)
        if _state['scrub']:
            return

        # Finalizar una sola vez cuando la sim termine
        if _state['done'] and not _state['finalized']:
            _state['finalized'] = True
            _final_render()
            return

        n = len(_raw)
        if n < 2:
            pct_sim = 0.0
            info.set_text('Simulando… 0%')
            return

        # Punto actual = último dato disponible
        last = _raw[-1]
        t_l, Vl, Il, Pl, Dl = last[0], last[1], last[2], last[3], last[4]
        Gl   = irr_fn(t_l / 1e3)

        # Trail en curva P-V
        j0 = max(0, n - TRAIL)
        tv = [_raw[k][1] for k in range(j0, n)]
        tp = [_raw[k][3] for k in range(j0, n)]

        # Historia submuestreada para líneas de tiempo
        step = max(1, n // RT_DISPLAY)
        hist = _raw[::step]
        arr  = np.array(hist)
        t_h  = arr[:, 0];  V_h = arr[:, 1];  I_h = arr[:, 2]
        P_h  = arr[:, 3];  D_h = arr[:, 4]
        G_h  = np.array([irr_fn(t / 1e3) for t in t_h])

        dot_pv.set_data([Vl], [Pl])
        trail_pv.set_data(tv, tp)
        dot_iv.set_data([Vl], [Il])
        line_vt.set_data(t_h, V_h);   dot_vt.set_data([t_l], [Vl])
        line_pt.set_data(t_h, P_h);   dot_pt.set_data([t_l], [Pl])
        line_it.set_data(t_h, I_h);   dot_it.set_data([t_l], [Il])
        line_dt.set_data(t_h, D_h);   dot_dt.set_data([t_l], [Dl])
        line_gt.set_data(t_h, G_h);   dot_gt.set_data([t_l], [Gl])

        pct_sim = n / n_total * 100
        pct_mpp = Pl / Pmpp * 100 if Pmpp > 0 else 0
        info.set_text(
            f'Simulando… {pct_sim:.0f}%  |  t = {t_l:.0f} ms  |  '
            f'Vpv = {Vl:.2f} V  |  Ppv = {Pl:.1f} W  |  {pct_mpp:.1f}% MPP')

    # ── Slider (desactivado hasta que termine la sim) ─────────────────────────
    ax_sl  = fig.add_axes([0.10, 0.095, 0.80, 0.022], facecolor='#1a1a2e')
    slider = Slider(ax_sl, 't', 0, 1, valinit=0, valstep=1, color=C_CURVE)
    slider.label.set_color('white')
    slider.valtext.set_color(C_DOT)
    slider.set_active(False)

    def _on_slider(val):
        if _state['done']:
            _state['scrub'] = True
            draw_scrub(int(val))
    slider.on_changed(_on_slider)

    # ── Botones (desactivados hasta terminar la sim) ──────────────────────────
    kw_btn = dict(color='#2a2a3a', hovercolor='#3a3a5e')
    ax_rst  = fig.add_axes([0.35, 0.038, 0.07, 0.030], facecolor='#2a2a3a')
    ax_play = fig.add_axes([0.43, 0.038, 0.07, 0.030], facecolor='#2a2a3a')
    ax_pau  = fig.add_axes([0.51, 0.038, 0.07, 0.030], facecolor='#2a2a3a')
    btn_rst  = Button(ax_rst,  '↺ Reset', **kw_btn)
    btn_play = Button(ax_play, '▶ Play',  **kw_btn)
    btn_pau  = Button(ax_pau,  '⏸ Pausa', **kw_btn)
    for btn in (btn_rst, btn_play, btn_pau):
        btn.label.set_color('#666688')
        btn.set_active(False)

    play_state = {'running': False, 'frame': 0}

    def _anim_step(_):
        if play_state['running'] and _state['done']:
            n = len(_raw) - 1
            play_state['frame'] = min(play_state['frame'] + 1, n)
            slider.set_val(play_state['frame'])

    def _on_play(_):
        if not _state['done']:
            return
        _state['scrub']         = True
        play_state['running']   = True

    def _on_pause(_):
        play_state['running'] = False

    def _on_reset(_):
        play_state['running'] = False
        play_state['frame']   = 0
        _state['scrub']       = True
        slider.set_val(0)

    btn_play.on_clicked(_on_play)
    btn_pau.on_clicked(_on_pause)
    btn_rst.on_clicked(_on_reset)

    # ── Hilo worker: corre la simulación en paralelo ──────────────────────────
    def _sim_worker():
        for _, t, Vpv, Ipv, Ppv, D, Vref, Vco in engine.iter_steps():
            _raw.append((t * 1e3, Vpv, Ipv, Ppv, D, Vref, Vco))
        _state['done'] = True

    threading.Thread(target=_sim_worker, daemon=True).start()

    # ── FuncAnimation: actualiza cada 30ms ────────────────────────────────────
    ani = _mplanim.FuncAnimation(   # noqa: F841
        fig, _update,
        interval=30,
        blit=False,
        cache_frame_data=False)

    # FuncAnimation también avanza el play del slider
    _orig_update = ani._step

    def _combined_step(*a):
        _anim_step(None)
        return _orig_update(*a)

    ani._step = _combined_step

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Animación MPPT en tiempo real')
    ap.add_argument('--algo', choices=['po', 'inc', 'pso'], default='po')
    ap.add_argument('--irr',  choices=['constant', 'partial_shading'],
                    default='constant')
    args = ap.parse_args()
    launch(algo=args.algo, irr=args.irr)
