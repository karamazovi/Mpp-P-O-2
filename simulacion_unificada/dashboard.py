"""
dashboard.py — Dashboard interactivo con simulación en TIEMPO REAL.

Variables mostradas (cuadro azul Simulink):
  Panel 1: Vpv [V]  +  Vref overlay
  Panel 2: Ppv [W]  +  línea Pref=85W
  Panel 3: Ipv [A]
  Panel 4: Vref [V] standalone

Los 4 paneles comparten eje X → zoom/pan sincronizados.
Al finalizar la simulación se renderiza con resolución completa.

Ejecutar: python dashboard.py
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from simulation_engine import SimulationEngine
import animation as _anim_mod

# ── Paleta ────────────────────────────────────────────────────────────────────
BG     = '#1e1e2e'
PANEL  = '#181825'
BORDER = '#45475a'
TEXT   = '#cdd6f4'
BLUE   = '#89b4fa'
GREEN  = '#a6e3a1'
RED    = '#f38ba8'
PEACH  = '#fab387'
PURPLE = '#cba6f7'
YELLOW = '#f9e2af'
WHITE  = '#ffffff'

DEFAULT_BOOST = dict(L=330e-6, Ci=22e-6, Co=22e-6,
                     RL=60e-3, Ron=35e-3, RB=69e-3, RCo=6e-3)
DEFAULT_PANEL = dict(cells_in_series=36, isc_ref=5.0, voc_ref=22.2,
                     vmpp_ref=18.1, impp_ref=4.7,
                     ideality_factor=1.3, series_resistance=0.221)

PREF_W = 85.07          # 18.1 V × 4.7 A

# Puntos mostrados en tiempo real (submuestreo solo para velocidad de dibujo)
RT_DISPLAY = 6_000
# Puntos mostrados en render final (todos los 120k pasos → igual que MATLAB)
FINAL_DISPLAY = 120_000   # matplotlib lo aguanta sin problema


# ─────────────────────────────────────────────────────────────────────────────
class Dashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Emulador PV Unificado — Tiempo Real")
        self.geometry("1500x920")
        self.configure(bg=BG)

        self._sim_queue: queue.Queue = queue.Queue(maxsize=4)
        self._running   = False
        self._engine: SimulationEngine | None = None

        # Lista compartida: el worker hace append() con GIL → thread-safe.
        # Acumula TODOS los pasos (120k para 2.4s, dt=20µs) — igual que MATLAB.
        self._raw: list = []   # cada elemento: (t_ms, Vpv, Ipv, Ppv, D, Vref, Vco2, Vco1)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        for w in ('TFrame', 'TLabel', 'TCheckbutton', 'TRadiobutton',
                  'TLabelframe', 'TLabelframe.Label'):
            style.configure(w, background=BG, foreground=TEXT)
        style.configure('TEntry', fieldbackground=PANEL, foreground=TEXT)

        # ── Sidebar con scroll ────────────────────────────────────────────────
        side_outer = tk.Frame(self, width=310, bg=BG)
        side_outer.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        side_outer.pack_propagate(False)

        # Canvas + scrollbar para sidebar scrollable
        _canvas = tk.Canvas(side_outer, bg=BG, highlightthickness=0, width=295)
        _sb     = ttk.Scrollbar(side_outer, orient='vertical',
                                command=_canvas.yview)
        _canvas.configure(yscrollcommand=_sb.set)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(_canvas)
        _win = _canvas.create_window((0, 0), window=side, anchor='nw')

        def _on_resize(event):
            _canvas.itemconfig(_win, width=event.width)
        _canvas.bind('<Configure>', _on_resize)

        def _on_frame(e):
            _canvas.configure(scrollregion=_canvas.bbox('all'))
        side.bind('<Configure>', _on_frame)

        def _on_wheel(e):
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        _canvas.bind_all('<MouseWheel>', _on_wheel)

        # ── MPPT ──────────────────────────────────────────────────────────────
        f = ttk.LabelFrame(side, text='Algoritmo MPPT', padding=6)
        f.pack(fill=tk.X, pady=3, padx=4)
        self.mppt_var = tk.StringVar(value='po')
        for val, lbl in [('po',  'P&O  (Perturb & Observe)'),
                          ('inc', 'INC  (Conductancia Incremental)'),
                          ('pso', 'PSO  (Particle Swarm)')]:
            ttk.Radiobutton(f, text=lbl, variable=self.mppt_var,
                            value=val).pack(anchor=tk.W)

        # ── Irradiancia ───────────────────────────────────────────────────────
        f_irr = ttk.LabelFrame(side, text='Irradiancia G', padding=6)
        f_irr.pack(fill=tk.X, pady=3, padx=4)
        self.irr_var = tk.StringVar(value='custom')

        ttk.Radiobutton(f_irr, text='G constante personalizada',
                        variable=self.irr_var, value='custom').pack(anchor=tk.W)
        row_g = ttk.Frame(f_irr); row_g.pack(fill=tk.X, pady=2)
        ttk.Label(row_g, text='G [W/m²]', width=10).pack(side=tk.LEFT)
        self.G_var = tk.StringVar(value='1000')
        ttk.Entry(row_g, textvariable=self.G_var, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Radiobutton(f_irr, text='Sombreado parcial (G→100 en 4 eventos)',
                        variable=self.irr_var, value='partial_shading').pack(anchor=tk.W)

        # ── Panel PV ──────────────────────────────────────────────────────────
        fp = ttk.LabelFrame(side, text='Panel PV', padding=6)
        fp.pack(fill=tk.X, pady=3, padx=4)
        self._pvars = {}
        pv_fields = [
            ('Ns (celdas)',  'cells_in_series', 36,    ''),
            ('Isc [A]',      'isc_ref',          5.0,   ''),
            ('Voc [V]',      'voc_ref',          22.2,  ''),
            ('Vmpp [V]',     'vmpp_ref',         18.1,  ''),
            ('Impp [A]',     'impp_ref',         4.7,   ''),
            ('n (idealidad)','ideality_factor',  1.3,   ''),
            ('Rs [Ω]',       'series_resistance',0.221, ''),
        ]
        for lbl, key, default, _ in pv_fields:
            row = ttk.Frame(fp); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=lbl, width=13).pack(side=tk.LEFT)
            v = tk.StringVar(value=str(default))
            self._pvars[key] = v
            ttk.Entry(row, textvariable=v, width=8).pack(side=tk.LEFT, padx=2)
        # Temperatura de operación (afecta curva P-V y simulación)
        row_t = ttk.Frame(fp); row_t.pack(fill=tk.X, pady=1)
        ttk.Label(row_t, text='T [°C]', width=13).pack(side=tk.LEFT)
        self._pvars['T'] = tk.StringVar(value='25')
        ttk.Entry(row_t, textvariable=self._pvars['T'], width=8).pack(side=tk.LEFT, padx=2)

        # ── Control interno ───────────────────────────────────────────────────
        f2 = ttk.LabelFrame(side, text='Controlador interno', padding=6)
        f2.pack(fill=tk.X, pady=3, padx=4)
        self.use_mpc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f2, text='MPC analítico',
                        variable=self.use_mpc_var).pack(anchor=tk.W)
        ttk.Label(f2, text='P&O + MPC → perturba Vref  |  Sin MPC → steady-state D',
                  font=('Arial', 8), foreground=BORDER).pack(anchor=tk.W)

        # ── Boost ─────────────────────────────────────────────────────────────
        f3 = ttk.LabelFrame(side, text='Convertidor Boost', padding=6)
        f3.pack(fill=tk.X, pady=3, padx=4)
        self._bvars = {}
        for lbl, key, default in [('L [µH]','L',330), ('Ci [µF]','Ci',22),
                                   ('Co [µF]','Co',22), ('RL [mΩ]','RL',60),
                                   ('Ron [mΩ]','Ron',35), ('RB [mΩ]','RB',69),
                                   ('RCo [mΩ]','RCo',6),
                                   ('VB [V]','VB',24)]:
            row = ttk.Frame(f3); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=lbl, width=11).pack(side=tk.LEFT)
            v = tk.StringVar(value=str(default))
            self._bvars[key] = v
            ttk.Entry(row, textvariable=v, width=9).pack(side=tk.LEFT, padx=2)

        # ── Simulación ────────────────────────────────────────────────────────
        f4 = ttk.LabelFrame(side, text='Simulación', padding=6)
        f4.pack(fill=tk.X, pady=3, padx=4)
        self.ttotal_var = tk.StringVar(value='2.4')
        self.tmppt_var  = tk.StringVar(value='20')
        self.deltad_var = tk.StringVar(value='0.02')
        for lbl, v in [('T total [s]', self.ttotal_var),
                        ('T_MPPT [ms]', self.tmppt_var),
                        ('Δd P&O',      self.deltad_var)]:
            row = ttk.Frame(f4); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=lbl, width=13).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=v, width=8).pack(side=tk.LEFT, padx=2)

        # ── Botones ───────────────────────────────────────────────────────────
        tk.Button(side, text='Actualizar curva P-V',
                  bg='#313244', fg=TEXT, font=('Arial', 9),
                  command=self._draw_pv_curve).pack(fill=tk.X, pady=2, padx=4)

        tk.Button(side, text='▶  SIMULAR',
                  bg=GREEN, fg=BG, font=('Arial', 11, 'bold'),
                  command=self._start).pack(fill=tk.X, pady=4, padx=4)
        tk.Button(side, text='⏹  DETENER',
                  bg=RED, fg=BG, font=('Arial', 10, 'bold'),
                  command=self._stop).pack(fill=tk.X, pady=2, padx=4)
        tk.Button(side, text='Comparar 3 algoritmos',
                  bg=BLUE, fg=BG, font=('Arial', 9, 'bold'),
                  command=self._compare).pack(fill=tk.X, pady=2, padx=4)
        tk.Button(side, text='Exportar CSV',
                  bg=PEACH, fg=BG, font=('Arial', 9),
                  command=self._export).pack(fill=tk.X, pady=2, padx=4)
        tk.Button(side, text='Validar vs Simulink CSV',
                  bg=PURPLE, fg=BG, font=('Arial', 9),
                  command=self._validate).pack(fill=tk.X, pady=2, padx=4)
        tk.Button(side, text='Animación paso a paso',
                  bg='#6c8ebf', fg=BG, font=('Arial', 9, 'bold'),
                  command=self._open_animation).pack(fill=tk.X, pady=2, padx=4)

        self.status_var = tk.StringVar(value='Listo.')
        ttk.Label(side, textvariable=self.status_var,
                  wraplength=280, font=('Arial', 9)).pack(pady=4, padx=4)
        self.progress = ttk.Progressbar(side, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=4, pady=(0, 8))

        # ── Gráficas ──────────────────────────────────────────────────────────
        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.fig = Figure(figsize=(13, 8.5), facecolor=BG)
        from matplotlib.gridspec import GridSpec as _GS
        _gs = _GS(4, 3, figure=self.fig,
                  hspace=0.52, wspace=0.35,
                  left=0.06, right=0.97, top=0.95, bottom=0.06)

        # ── Curva P-V (columna izquierda, ocupa las 4 filas) ─────────────────
        ax_pv = self.fig.add_subplot(_gs[:, 0])
        self._style_ax(ax_pv, 'Curva P-V  —  punto de operación')
        ax_pv.set_xlabel('Vpv  [V]', color=TEXT, fontsize=7)
        ax_pv.set_ylabel('Ppv  [W]', color=TEXT, fontsize=7)

        # Curva P-V estática (se redibuja con _draw_pv_curve)
        self._ln_pvcurve, = ax_pv.plot([], [], color='#4fc3f7', lw=1.8,
                                        label='Curva P-V')
        # Línea vertical MPP teórico
        self._ln_vmpp = ax_pv.axvline(0, color=RED, lw=0.9, ls=':', alpha=0.8)
        # Triángulo en el MPP
        self._dot_mpp, = ax_pv.plot([], [], 'v', color=RED, ms=9, zorder=5,
                                     label='MPP teórico')
        # Punto de operación actual (se mueve durante la simulación)
        self._dot_op,  = ax_pv.plot([], [], 'o', color='#ffd54f', ms=9,
                                     zorder=6, label='Op. actual')
        # Línea vertical Vpv actual
        self._ln_vpv_v = ax_pv.axvline(0, color='#ffd54f', lw=0.8,
                                        ls='--', alpha=0.6)
        ax_pv.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                     loc='upper right')
        self._ax_pv = ax_pv

        # ── Series de tiempo (columnas 1-2, 4 filas) ─────────────────────────
        ax0 = self.fig.add_subplot(_gs[0, 1:])
        ax1 = self.fig.add_subplot(_gs[1, 1:], sharex=ax0)
        ax2 = self.fig.add_subplot(_gs[2, 1:], sharex=ax0)
        ax3 = self.fig.add_subplot(_gs[3, 1:], sharex=ax0)
        self._ax = [ax0, ax1, ax2, ax3]

        for ax, title in zip(self._ax,
                             ['Vpv  [V]', 'Ppv  [W]  /  Vco  [V]',
                              'Ipv  [A]', 'D  [duty cycle]']):
            self._style_ax(ax, title)
        ax3.set_xlabel('t  [ms]', color=TEXT, fontsize=7)

        # Márgenes automáticos por eje — auto-zoom independiente
        for ax in self._ax:
            ax.margins(y=0.10)
            ax.set_autoscaley_on(True)

        # Vpv + Vmpp teórico (línea horizontal) + Vref MPPT (solo INC/PSO)
        self._ln_vpv,  = ax0.plot([], [], color=GREEN,  lw=0.7, label='Vpv')
        # Vmpp teórico: línea horizontal constante (objetivo real del MPPT)
        self._ln_vmpp_t = ax0.axhline(18.1, color=RED, lw=1.0,
                                       linestyle='--', alpha=0.85,
                                       label='Vmpp teórico')
        # Vref del algoritmo (útil en INC/PSO; en P&O = copia de Vpv → se oculta)
        self._ln_vref, = ax0.plot([], [], color='#f9e2af', lw=0.6,
                                   linestyle=':', alpha=0.7, label='Vref MPPT')
        ax0.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, loc='upper right')

        # Ppv + Pref  +  Vco en eje Y derecho (twin)
        self._ln_ppv,  = ax1.plot([], [], color=YELLOW, lw=0.7, label='Ppv')
        self._ln_pref  = ax1.axhline(PREF_W, color=WHITE, lw=0.9,
                                      linestyle='--', alpha=0.8,
                                      label=f'Pref = {PREF_W:.1f} W')
        self._ax1_twin = ax1.twinx()
        self._style_ax(self._ax1_twin, '')
        self._ax1_twin.set_ylabel('Vco  [V]', color=PURPLE, fontsize=7)
        self._ax1_twin.tick_params(colors=PURPLE, labelsize=7)
        self._ax1_twin.margins(y=0.12)
        self._ax1_twin.set_autoscaley_on(True)
        self._ln_vco, = self._ax1_twin.plot([], [], color=PURPLE, lw=0.7,
                                              linestyle='--', alpha=0.85, label='Vco')
        # Leyenda combinada ax1 + twin
        lns = [self._ln_ppv, self._ln_pref, self._ln_vco]
        ax1.legend(lns, [l.get_label() for l in lns],
                   fontsize=6, facecolor=PANEL, labelcolor=TEXT, loc='upper right')

        # Ipv
        self._ln_ipv,  = ax2.plot([], [], color=PEACH,  lw=0.7, label='Ipv')
        ax2.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, loc='upper right')

        # D (duty cycle) — antes era Vref redundante
        self._ln_d, = ax3.plot([], [], color=PURPLE, lw=0.7, label='D')
        ax3.set_ylim(0, 1)
        ax3.set_autoscaley_on(False)   # D siempre en [0,1]
        ax3.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, loc='upper right')

        self._canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar con zoom, pan, home, guardar figura
        toolbar = NavigationToolbar2Tk(self._canvas, fig_frame)
        toolbar.config(background=BG)
        toolbar.update()

        # Dibujar curva P-V inicial con parámetros por defecto
        self._draw_pv_curve()

    def _style_ax(self, ax, title: str):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.set_title(title, color=BLUE, fontsize=8, pad=2, loc='left')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.5)

    def _draw_pv_curve(self, G: float = None):
        """
        Dibuja/actualiza la curva P-V con los parámetros del panel actuales.
        Marca el MPP teórico con triángulo rojo.
        Se llama al inicio y cada vez que cambian los parámetros del panel o G.
        """
        from pv_panel import PanelPV
        pp = self._get_panel_params()
        try:
            panel = PanelPV(pp)
        except Exception:
            return

        # G para la curva: si no se especifica, leer del campo
        if G is None:
            try:
                G = float(self.G_var.get())
            except ValueError:
                G = 1000.0
            G = max(1.0, min(1500.0, G))

        T_op = self._get_temperatura()
        V_arr, _, P_arr = panel.curva_pv(G, T_op, n_puntos=400)
        Vmpp, Impp, Pmpp    = panel.mpp(G, T_op)

        # Actualizar curva
        self._ln_pvcurve.set_data(V_arr, P_arr)
        # Actualizar marcadores MPP
        self._ln_vmpp.set_xdata([Vmpp, Vmpp])
        self._dot_mpp.set_data([Vmpp], [Pmpp])

        # Ajustar límites de ejes
        ax = self._ax_pv
        ax.set_xlim(0, V_arr[-1] * 1.05)
        ax.set_ylim(0, P_arr.max() * 1.18)

        # Actualizar leyenda con valores numéricos
        self._dot_mpp.set_label(
            f'MPP teórico\n({Vmpp:.2f} V, {Pmpp:.1f} W)\nImpp={Impp:.3f} A')
        ax.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, loc='upper right')

        # Guardar Vmpp/Pmpp para usar en _update_plot
        self._vmpp_teorico = Vmpp
        self._pmpp_teorico = Pmpp

        self._canvas.draw_idle()

    # ── Engine ────────────────────────────────────────────────────────────────
    def _get_panel_params(self) -> dict:
        types = dict(cells_in_series=int)   # Ns es entero, resto float
        _skip = {'T'}                        # temperatura: parámetro operativo, no de panel
        out   = {}
        for key, v in self._pvars.items():
            if key in _skip:
                continue
            try:
                val = types.get(key, float)(v.get())
            except ValueError:
                val = DEFAULT_PANEL[key]
            out[key] = val
        return out

    def _get_irr_profile(self):
        """Retorna 'partial_shading' o un callable G=cte según la selección."""
        if self.irr_var.get() == 'partial_shading':
            return 'partial_shading'
        # G constante personalizada
        try:
            G = float(self.G_var.get())
        except ValueError:
            G = 1000.0
        G = max(0.0, min(1500.0, G))
        return lambda _: G   # callable compatible con irr_fn del engine

    def _get_boost_params(self) -> dict:
        scale = dict(L=1e-6, Ci=1e-6, Co=1e-6, RL=1e-3, Ron=1e-3, RB=1e-3, RCo=1e-3)
        out = {}
        for k, v in self._bvars.items():
            if k == 'VB':
                continue   # VB no va al BoostConverter
            try:
                out[k] = float(v.get()) * scale.get(k, 1.0)
            except ValueError:
                out[k] = DEFAULT_BOOST.get(k, 0.0)
        return out

    def _get_VB(self) -> float:
        try:
            return float(self._bvars['VB'].get())
        except (ValueError, KeyError):
            return 24.0

    def _get_temperatura(self) -> float:
        try:
            return float(self._pvars['T'].get())
        except (ValueError, KeyError):
            return 25.0

    def _build_engine(self, algo: str, irr_profile=None) -> SimulationEngine:
        irr = irr_profile if irr_profile is not None else self._get_irr_profile()
        try:
            delta_d = float(self.deltad_var.get())
        except ValueError:
            delta_d = 0.02
        try:
            T_mppt = float(self.tmppt_var.get()) * 1e-3
        except ValueError:
            T_mppt = None

        mppt_params = {
            'po':  {'delta_d': delta_d},
            'inc': {'delta': 0.5},
            'pso': {'n_particles': 10, 'n_iter': 20},
        }[algo]

        return SimulationEngine(
            mppt_algo    = algo,
            use_mpc      = self.use_mpc_var.get(),
            panel_params = self._get_panel_params(),
            boost_params = self._get_boost_params(),
            mppt_params  = mppt_params,
            T_total      = float(self.ttotal_var.get()),
            T_mppt       = T_mppt,
            irr_profile  = irr,
            VB           = self._get_VB(),
            temperatura  = self._get_temperatura(),
        )

    # ── Simulación ────────────────────────────────────────────────────────────
    def _reset_axes(self):
        T_ms = float(self.ttotal_var.get()) * 1e3
        for ln in (self._ln_vpv, self._ln_vref, self._ln_ppv,
                   self._ln_ipv, self._ln_vco, self._ln_d):
            ln.set_data([], [])
        self._ax[0].set_xlim(0, T_ms)
        self._canvas.draw_idle()

    def _start(self):
        if self._running:
            return
        self._running = True
        self._raw.clear()
        self._reset_axes()
        self.progress.start(12)
        self.status_var.set('Simulando…')

        algo = self.mppt_var.get()
        self._engine = self._build_engine(algo)

        # Redibujar curva P-V con los parámetros actuales del panel y G elegida
        try:
            G_curva = float(self.G_var.get()) if self.irr_var.get() != 'partial_shading' else 1000.0
        except ValueError:
            G_curva = 1000.0
        self._draw_pv_curve(G=G_curva)

        # Actualizar Vmpp teórico (línea horizontal) con los parámetros del panel
        vmpp = self._engine.Vmpp
        self._ln_vmpp_t.set_ydata([vmpp, vmpp])
        self._ln_vmpp_t.set_label(f'Vmpp teórico = {vmpp:.2f} V')
        # En P&O, Vref = copia de Vpv → no aporta info extra → ocultar
        is_po = (self.mppt_var.get() == 'po')
        self._ln_vref.set_visible(not is_po)
        self._ax[0].legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                           loc='upper right')

        # Actualizar línea Pref con el Vmpp*Impp real del panel configurado
        pref = self._engine.Pref
        self._ln_pref.set_ydata([pref, pref])
        self._ln_pref.set_label(f'Pref = {pref:.1f} W')
        self._ax[1].legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                           loc='upper right')

        def _sim_worker():
            raw = self._raw          # referencia local → append() es O(1) amortizado
            for _, t, Vpv, Ipv, Ppv, D, Vref, Vco2, Vco1 in self._engine.iter_steps():
                if not self._running:
                    break
                # Guardar TODOS los pasos — 120k pts para 2.4s (igual que MATLAB)
                # Formato: (t_ms, Vpv, Ipv, Ppv, D, Vref, Vco2, Vco1)
                raw.append((t * 1e3, Vpv, Ipv, Ppv, D, Vref, Vco2, Vco1))
            # Señal de fin (no data)
            self._sim_queue.put(None)

        threading.Thread(target=_sim_worker, daemon=True).start()
        self._poll_queue()

    def _stop(self):
        self._running = False
        self.status_var.set('Detenido.')
        self.progress.stop()

    # ── Polling ───────────────────────────────────────────────────────────────
    def _poll_queue(self):
        # Verificar si llegó la señal de fin
        try:
            item = self._sim_queue.get_nowait()
            if item is None:
                self._running = False
                self.progress.stop()
                n = len(self._raw)
                self.status_var.set(
                    f'Completado  {n:,} muestras — zoom 🔍 / pan ✋')
                self._update_plot(final=True)
                return
        except queue.Empty:
            pass

        # Actualizar gráficas con los datos acumulados hasta ahora
        self._update_plot(final=False)

        if self._running:
            self.after(30, self._poll_queue)

    def _update_plot(self, final: bool = False):
        raw = self._raw
        n   = len(raw)
        if n < 2:
            return

        # ── Submuestreo ───────────────────────────────────────────────────────
        if final:
            step = 1
        else:
            step = max(1, n // RT_DISPLAY)
        data = raw[::step]

        # Conversión a numpy — una sola vez por frame
        arr  = np.array(data, dtype=np.float32)   # float32 → mitad de memoria
        t_ms = arr[:, 0]
        Vpv  = arr[:, 1]
        Ipv  = arr[:, 2]
        Ppv  = arr[:, 3]
        D    = arr[:, 4]
        Vref = arr[:, 5]
        Vco  = arr[:, 6]   # Vco2: tensión condensador
        # arr[:,7] = Vco1 (tensión terminal batería ≈ VB, constante)
        T_ms = float(self.ttotal_var.get()) * 1e3

        # ── Eje X compartido ──────────────────────────────────────────────────
        x_max = max(float(t_ms[-1]), T_ms)
        self._ax[0].set_xlim(0, x_max)

        # ── ax0: Vpv + Vref — auto-zoom Y independiente ───────────────────────
        self._ln_vpv.set_data(t_ms, Vpv)
        self._ln_vref.set_data(t_ms, Vref)
        self._ax[0].relim()
        self._ax[0].autoscale_view(scalex=False, scaley=True)

        # ── ax1: Ppv + Vco — auto-zoom Y por cada eje ─────────────────────────
        self._ln_ppv.set_data(t_ms, Ppv)
        self._ax[1].relim()
        self._ax[1].autoscale_view(scalex=False, scaley=True)
        self._ln_vco.set_data(t_ms, Vco)
        self._ax1_twin.relim()
        self._ax1_twin.autoscale_view(scalex=False, scaley=True)

        # ── ax2: Ipv — auto-zoom Y ────────────────────────────────────────────
        self._ln_ipv.set_data(t_ms, Ipv)
        self._ax[2].relim()
        self._ax[2].autoscale_view(scalex=False, scaley=True)

        # ── ax3: D (duty cycle) — rango fijo [0,1] ────────────────────────────
        self._ln_d.set_data(t_ms, D)

        # ── Punto de operación sobre la curva P-V ─────────────────────────────
        Vpv_last = float(Vpv[-1])
        Ppv_last = float(Ppv[-1])
        self._dot_op.set_data([Vpv_last], [Ppv_last])
        self._ln_vpv_v.set_xdata([Vpv_last, Vpv_last])

        # Leyenda P-V: solo se actualiza al final (costosa, innecesaria en RT)
        if final:
            vmpp = getattr(self, '_vmpp_teorico', 18.1)
            pmpp = getattr(self, '_pmpp_teorico', PREF_W)
            err  = abs(Vpv_last - vmpp)
            pct  = Ppv_last / pmpp * 100 if pmpp > 0 else 0
            D_last = float(D[-1])
            eta  = (Ppv_last / (float(self._get_VB()) * float(Ipv[-1]) *
                                (1.0 - D_last) + 1e-9)) * 100 \
                   if float(Ipv[-1]) * (1.0 - D_last) > 0 else float('nan')
            self._dot_op.set_label(
                f'Op. actual  ({Vpv_last:.2f} V, {Ppv_last:.1f} W)\n'
                f'|ΔV| = {err:.2f} V  —  {pct:.1f}% Pmpp')
            self._ax_pv.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                               loc='upper right')
            self._canvas.draw()       # bloqueante — render completo
        else:
            self._canvas.draw_idle()  # asíncrono — sin rebuild de leyendas

    # ── Comparar ──────────────────────────────────────────────────────────────
    def _compare(self):
        if self._running:
            messagebox.showinfo('Espera', 'Hay una simulación en curso.')
            return
        self.status_var.set('Comparando P&O / INC / PSO…')
        self.progress.start(12)
        irr = self.irr_var.get()

        def _worker():
            palette = {'po': RED, 'inc': PEACH, 'pso': GREEN}

            def _clear():
                for ax in self._ax:
                    # conservar línea Pref
                    for line in list(ax.lines):
                        if line.get_linestyle() != '--' or ax is not self._ax[1]:
                            line.remove()
                self._canvas.draw_idle()
            self.after(0, _clear)

            for algo, color in palette.items():
                e = self._build_engine(algo, irr_profile=irr)
                r = e.run()
                t_ms = r['t'] * 1e3
                n    = len(t_ms)
                step = max(1, n // RT_DISPLAY)
                sl   = slice(0, n, step)

                def _plot(c=color, a=algo, tm=t_ms[sl], rv=r, s=sl):
                    self._ax[0].plot(tm, rv['Vpv'][s],  color=c, lw=0.7,
                                     label=a.upper(), alpha=0.9)
                    self._ax[0].plot(tm, rv['Vref'][s], color=c, lw=0.5,
                                     linestyle='--', alpha=0.5)
                    self._ax[1].plot(tm, rv['Ppv'][s],  color=c, lw=0.7,
                                     label=a.upper(), alpha=0.9)
                    self._ax[2].plot(tm, rv['Ipv'][s],  color=c, lw=0.7,
                                     label=a.upper(), alpha=0.9)
                    self._ax[3].plot(tm, rv['D'][s], color=c, lw=0.7,
                                     label=a.upper(), alpha=0.9)
                    for ax in self._ax:
                        ax.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT)
                    self._canvas.draw_idle()

                self.after(0, _plot)

            self.after(0, self.progress.stop)
            self.after(0, lambda: self.status_var.set(
                'Comparación lista — zoom con 🔍, pan con ✋'))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Animación ─────────────────────────────────────────────────────────────
    def _open_animation(self):
        if self._running:
            messagebox.showinfo('Espera', 'Hay una simulación en curso.')
            return
        # Siempre lanzar tiempo real: crear engine fresco (sin results)
        algo = self.mppt_var.get()
        irr  = self.irr_var.get()
        eng  = self._build_engine(algo, irr_profile=irr)
        self.status_var.set('Abriendo animación en tiempo real…')
        # Lanzar en hilo separado para no bloquear el dashboard
        threading.Thread(
            target=lambda: _anim_mod.launch(eng),
            daemon=True).start()

    # ── Exportar / Validar ────────────────────────────────────────────────────
    def _export(self):
        if not self._engine or not self._engine.results:
            messagebox.showwarning('Sin datos', 'Ejecuta una simulación primero.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            initialfile='simulacion_pv.csv')
        if path:
            self._engine.export_csv(path)
            self.status_var.set(f'CSV: {os.path.basename(path)}')

    def _validate(self):
        if not self._engine or not self._engine.results:
            messagebox.showwarning('Sin datos', 'Ejecuta una simulación primero.')
            return
        path = filedialog.askopenfilename(
            title='CSV de referencia Simulink', filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            r   = self._engine.compare_simulink(path)
            msg = (f"∫Ppv sim : {r['integral_sim']:.4f} W·s\n"
                   f"∫Ppv ref : {r['integral_ref']:.4f} W·s\n"
                   f"Error    : {r['error_pct']:.2f}%\n\n"
                   f"{'✓ PASS (< 3%)' if r['passed'] else '✗ FAIL (≥ 3%)'}")
            messagebox.showinfo('Validación vs Simulink', msg)
        except Exception as exc:
            messagebox.showerror('Error', str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = Dashboard()
    app.mainloop()
