"""
Comparación P&O vs PSO+MPC vs MATLAB (referencia)
Carga los CSV de ambas simulaciones y genera gráficas comparativas.
Los datos MATLAB fueron extraídos visualmente de la gráfica del artículo.
Si se dispone del archivo matlab_resultados.csv, se carga automáticamente.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
CSV_PO  = os.path.join(BASE, 'simulacion_python',  'resultados.csv')
CSV_PSO = os.path.join(BASE, 'simulacion_pso_mpc', 'resultados_pso_mpc.csv')

# ── Cargar datos ───────────────────────────────────────────────────────────────
po  = pd.read_csv(CSV_PO)
pso = pd.read_csv(CSV_PSO)

t_po  = po['tiempo_s'].values  * 1e3   # → ms
t_pso = pso['tiempo_s'].values * 1e3

Vpv_po  = po['Vpv_V'].values;   Ppv_po  = po['Ppv_W'].values
Vpv_pso = pso['Vpv_V'].values;  Ppv_pso = pso['Ppv_W'].values
Ipv_po  = po['Ipv_A'].values;   Ipv_pso = pso['Ipv_A'].values
D_po    = po['D'].values;       D_pso   = pso['D'].values
G_po    = po['G_Wm2'].values;   G_pso   = pso['G_Wm2'].values

# Referencia PSO (solo disponible en ese CSV)
Vref_pso = pso['Vref_V'].values

# ── Datos MATLAB (referencia del artículo) ─────────────────────────────────────
# Se intenta cargar matlab_resultados.csv; si no existe, se usan datos
# aproximados extraídos visualmente de la gráfica del artículo.
CSV_MAT = os.path.join(BASE, 'matlab_resultados.csv')
if os.path.exists(CSV_MAT):
    mat = pd.read_csv(CSV_MAT)
    t_mat   = mat.iloc[:, 0].values * 1e3
    Vpv_mat = mat.iloc[:, 1].values
    Ppv_mat = mat.iloc[:, 2].values
    MATLAB_REAL = True
    print("✓ Datos MATLAB cargados desde matlab_resultados.csv")
else:
    # ── Reconstrucción aproximada a partir de la imagen del artículo ──────────
    # Segmentos extraídos visualmente (t en ms, Vpv en V, Ppv en W):
    #  0–290ms  : convergencia con oscilaciones grandes (18.8→18.1V, 90→83W)
    #  290–440ms: sombreado 1 → Vpv baja ~16.8V, Ppv → 0W
    #  440–880ms: recuperación → 18.1V / 83W estable
    #  880–990ms: sombreado 2
    #  990–1180ms: recuperación
    #  1180–1290ms: sombreado 3
    #  1290–1380ms: recuperación
    #  1380–1540ms: sombreado 4
    #  1540–1900ms: recuperación final → 18.1V / 83W
    def _seg(t0, t1, v0, v1, p0, p1, n=30):
        t = np.linspace(t0, t1, n)
        v = np.linspace(v0, v1, n)
        p = np.linspace(p0, p1, n)
        return t, v, p

    segments = [
        _seg(0,    50,   19.2, 18.5, 92,  88),
        _seg(50,   150,  18.5, 18.2, 88,  84),
        _seg(150,  290,  18.2, 18.1, 84,  83),
        _seg(290,  310,  18.1, 16.8, 83,   2),   # inicio sombreado 1
        _seg(310,  440,  16.8, 16.8,  2,   2),   # sombreado 1
        _seg(440,  480,  16.8, 18.0,  2,  80),   # fin sombreado 1
        _seg(480,  880,  18.0, 18.1, 80,  83),   # estable
        _seg(880,  900,  18.1, 16.8, 83,   2),   # inicio sombreado 2
        _seg(900,  990,  16.8, 16.8,  2,   2),
        _seg(990, 1010,  16.8, 18.0,  2,  80),
        _seg(1010,1180,  18.0, 18.1, 80,  83),
        _seg(1180,1200,  18.1, 16.8, 83,   2),   # inicio sombreado 3
        _seg(1200,1290,  16.8, 16.8,  2,   2),
        _seg(1290,1310,  16.8, 18.0,  2,  80),
        _seg(1310,1380,  18.0, 18.1, 80,  83),
        _seg(1380,1400,  18.1, 16.8, 83,   2),   # inicio sombreado 4
        _seg(1400,1540,  16.8, 16.8,  2,   2),
        _seg(1540,1580,  16.8, 18.0,  2,  80),
        _seg(1580,1900,  18.0, 18.1, 80,  83),
    ]
    t_mat   = np.concatenate([s[0] for s in segments])
    Vpv_mat = np.concatenate([s[1] for s in segments])
    Ppv_mat = np.concatenate([s[2] for s in segments])
    MATLAB_REAL = False
    print("⚠ matlab_resultados.csv no encontrado — usando aproximación visual de la imagen")

# ── Métricas ───────────────────────────────────────────────────────────────────
Pmpp = 85.07   # W  (referencia del artículo)
Vmpp = 18.1    # V

# Eficiencia de seguimiento promedio (solo zona estacionaria: t > 500ms)
mask_po  = t_po  > 500
mask_pso = t_pso > 500

eff_po  = np.mean(Ppv_po[mask_po])  / Pmpp * 100
eff_pso = np.mean(Ppv_pso[mask_pso]) / Pmpp * 100

# Tiempo de convergencia: primer instante donde Ppv supera 90% del máximo
# real simulado (no el teórico), solo durante zona de G=1000 W/m²
# Nota: P&O alcanza ~80W ≈ 94% del Pmpp teórico (85W), pero nunca 95%
# porque el modelo de diodo simple difiere ligeramente del panel real.
# Se usa 90% del máximo simulado para una comparación justa entre algoritmos.
Pmax_po  = np.max(Ppv_po [G_po  == 1000])   # máximo real alcanzado en G=1000
Pmax_pso = np.max(Ppv_pso[G_pso == 1000])
umbral_po  = 0.90 * Pmax_po
umbral_pso = 0.90 * Pmax_pso

def tiempo_convergencia(t, Ppv, G, umbral, ventana=10):
    """
    Primer instante (después de t=50ms) donde Ppv supera el umbral
    y se mantiene por encima durante al menos `ventana` muestras consecutivas.
    Solo se evalúa en períodos de G=1000 W/m².
    """
    inicio = np.searchsorted(t, 50)   # ignorar los primeros 50ms
    for i in range(inicio, len(t) - ventana):
        if G[i] == 1000 and np.all(Ppv[i:i+ventana] > umbral):
            return t[i]
    return float('nan')

conv_po  = tiempo_convergencia(t_po,  Ppv_po,  G_po,  umbral_po)
conv_pso = tiempo_convergencia(t_pso, Ppv_pso, G_pso, umbral_pso)

# Ripple en estado estacionario (desviación estándar de Ppv en zona estable)
ripple_po  = np.std(Ppv_po[mask_po])
ripple_pso = np.std(Ppv_pso[mask_pso])

print("=" * 52)
print(f"{'Métrica':<30} {'P&O':>8} {'PSO+MPC':>10}")
print("=" * 52)
print(f"{'Ppv promedio (t>500ms) [W]':<30} {np.mean(Ppv_po[mask_po]):>8.2f} {np.mean(Ppv_pso[mask_pso]):>10.2f}")
print(f"{'Eficiencia seguimiento [%]':<30} {eff_po:>8.1f} {eff_pso:>10.1f}")
print(f"{'Vpv promedio (t>500ms) [V]':<30} {np.mean(Vpv_po[mask_po]):>8.3f} {np.mean(Vpv_pso[mask_pso]):>10.3f}")
print(f"{'Tiempo convergencia [ms]':<30} {conv_po:>8.1f} {conv_pso:>10.1f}")
print(f"{'Ripple Ppv std [W]':<30} {ripple_po:>8.3f} {ripple_pso:>10.3f}")
print(f"{'D final':<30} {D_po[-1]:>8.4f} {D_pso[-1]:>10.4f}")
print("=" * 52)

# ── Estilo ─────────────────────────────────────────────────────────────────────
BG    = '#0f0f1a'
GRID  = '#2a2a3a'
C_PO  = '#4fc3f7'   # azul claro  → P&O
C_PSO = '#ff6b6b'   # rojo coral  → PSO+MPC
C_MAT = '#69f0ae'   # verde       → MATLAB referencia
C_REF = '#ffd54f'   # amarillo    → MPP teórico
MAT_LABEL = 'MATLAB (artículo)' if MATLAB_REAL else 'MATLAB (aprox. visual)'

plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white'})

# ══════════════════════════════════════════════════════════════════════════════
# Figura 1: Comparación temporal completa
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(13, 10))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Comparación P&O  vs  PSO+MPC', color='white', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(4, 1, hspace=0.45, left=0.08, right=0.97, top=0.92, bottom=0.07)

def estilo(ax, ylabel, ylims=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, linestyle='--')
    ax.set_ylabel(ylabel, color='#aaaacc', fontsize=9)
    if ylims:
        ax.set_ylim(*ylims)

ax1 = fig1.add_subplot(gs[0])
ax1.plot(t_mat, Vpv_mat, color=C_MAT, lw=1.4, ls='--', alpha=0.9, label=MAT_LABEL)
ax1.plot(t_po,  Vpv_po,  color=C_PO,  lw=1.2, label='P&O (Python)')
ax1.plot(t_pso, Vpv_pso, color=C_PSO, lw=1.2, label='PSO+MPC (Python)')
ax1.plot(t_pso, Vref_pso, color=C_REF, lw=0.8, ls=':', alpha=0.7, label='Vref PSO')
ax1.axhline(Vmpp, color='white', ls=':', lw=0.8, alpha=0.5)
estilo(ax1, 'Vpv (V)', (12, 24))
ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID, loc='lower right')

ax2 = fig1.add_subplot(gs[1], sharex=ax1)
ax2.plot(t_mat, Ppv_mat, color=C_MAT, lw=1.4, ls='--', alpha=0.9, label=MAT_LABEL)
ax2.plot(t_po,  Ppv_po,  color=C_PO,  lw=1.2, label='P&O (Python)')
ax2.plot(t_pso, Ppv_pso, color=C_PSO, lw=1.2, label='PSO+MPC (Python)')
ax2.axhline(Pmpp, color=C_REF, ls='--', lw=1.5, alpha=0.9,
            label=f'MPP teórico máximo = {Pmpp} W  (Vmpp={Vmpp}V, Impp=4.7A)')
ax2.text(t_po[-1]*0.01, Pmpp + 1.5, f'Pmpp teórico = {Pmpp} W',
         color=C_REF, fontsize=7.5, va='bottom')
estilo(ax2, 'Ppv (W)', (-5, 100))
ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID, loc='lower right')

ax3 = fig1.add_subplot(gs[2], sharex=ax1)
ax3.plot(t_po,  D_po,  color=C_PO,  lw=1.2, label='P&O')
ax3.plot(t_pso, D_pso, color=C_PSO, lw=1.2, label='PSO+MPC')
ax3.axhline(1 - Vmpp/24, color=C_REF, ls='--', lw=0.8, alpha=0.7, label=f'D_ss={1-Vmpp/24:.3f}')
estilo(ax3, 'Duty Cycle D', (0.1, 0.45))
ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

ax4 = fig1.add_subplot(gs[3], sharex=ax1)
ax4.plot(t_po,  G_po,  color=C_PO,  lw=1.5, alpha=0.9)
ax4.plot(t_pso, G_pso, color=C_PSO, lw=1.0, ls='--', alpha=0.7)
estilo(ax4, 'G (W/m²)', (-50, 1150))
ax4.set_xlabel('Tiempo (ms)', color='#aaaacc', fontsize=9)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

# ══════════════════════════════════════════════════════════════════════════════
# Figura 2: Zoom en transitorio inicial (0–600ms)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig2.patch.set_facecolor(BG)
fig2.suptitle('Zoom transitorio inicial (0–600ms)', color='white', fontsize=13, fontweight='bold')

mask_zoom_po  = t_po  <= 600
mask_zoom_pso = t_pso <= 600
mask_zoom_mat = t_mat <= 600

for ax in axes2:
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, linestyle='--')

axes2[0].plot(t_mat[mask_zoom_mat], Vpv_mat[mask_zoom_mat], color=C_MAT, lw=1.5, ls='--', label=MAT_LABEL)
axes2[0].plot(t_po[mask_zoom_po],   Vpv_po[mask_zoom_po],   color=C_PO,  lw=1.5, label='P&O (Python)')
axes2[0].plot(t_pso[mask_zoom_pso], Vpv_pso[mask_zoom_pso], color=C_PSO, lw=1.5, label='PSO+MPC (Python)')
axes2[0].axhline(Vmpp, color=C_REF, ls='--', lw=1, label=f'Vmpp={Vmpp}V')
axes2[0].set_ylabel('Vpv (V)', color='#aaaacc')
axes2[0].legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

axes2[1].plot(t_mat[mask_zoom_mat], Ppv_mat[mask_zoom_mat], color=C_MAT, lw=1.5, ls='--', label=MAT_LABEL)
axes2[1].plot(t_po[mask_zoom_po],   Ppv_po[mask_zoom_po],   color=C_PO,  lw=1.5, label='P&O (Python)')
axes2[1].plot(t_pso[mask_zoom_pso], Ppv_pso[mask_zoom_pso], color=C_PSO, lw=1.5, label='PSO+MPC (Python)')
axes2[1].axhline(Pmpp, color=C_REF, ls='--', lw=1.5,
                 label=f'MPP teórico máximo = {Pmpp} W')
axes2[1].text(5, Pmpp + 1.5, f'Pmpp teórico = {Pmpp} W',
              color=C_REF, fontsize=8, va='bottom')
axes2[1].set_ylabel('Ppv (W)', color='#aaaacc')
axes2[1].set_xlabel('Tiempo (ms)', color='#aaaacc')
axes2[1].legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', edgecolor=GRID)

fig2.tight_layout()

# ══════════════════════════════════════════════════════════════════════════════
# Figura 3: Tabla de métricas visual
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax_t = plt.subplots(figsize=(8, 3.5))
fig3.patch.set_facecolor(BG)
ax_t.set_facecolor(BG)
ax_t.axis('off')
fig3.suptitle('Resumen de métricas', color='white', fontsize=13, fontweight='bold')

filas = [
    ['MPP teórico (artículo)',   f'{Pmpp} W @ {Vmpp} V',               f'{Pmpp} W @ {Vmpp} V'],
    ['Ppv máx. simulado (G=1000)', f'{Pmax_po:.2f} W',                 f'{Pmax_pso:.2f} W'],
    ['Ppv promedio (t>500ms)',  f'{np.mean(Ppv_po[mask_po]):.2f} W',   f'{np.mean(Ppv_pso[mask_pso]):.2f} W'],
    ['Eficiencia vs MPP teórico', f'{eff_po:.1f} %',                   f'{eff_pso:.1f} %'],
    ['Vpv promedio (t>500ms)',  f'{np.mean(Vpv_po[mask_po]):.3f} V',   f'{np.mean(Vpv_pso[mask_pso]):.3f} V'],
    ['Conv. (90% del Pmpp sim.)', f'{conv_po:.1f} ms',                 f'{conv_pso:.1f} ms'],
    ['Ripple Ppv (std, t>500ms)', f'{ripple_po:.3f} W',                f'{ripple_pso:.3f} W'],
    ['D en estado estacionario', f'{np.mean(D_po[mask_po]):.4f}',      f'{np.mean(D_pso[mask_pso]):.4f}'],
]

mask_mat_ss = t_mat > 500
Ppv_mat_ss  = Ppv_mat[mask_mat_ss & (Ppv_mat > 5)]   # excluir sombreados
Vpv_mat_ss  = Vpv_mat[mask_mat_ss & (Ppv_mat > 5)]

filas_mat_ppv  = f'{np.mean(Ppv_mat_ss):.2f} W' if len(Ppv_mat_ss) else 'N/A'
filas_mat_vpv  = f'{np.mean(Vpv_mat_ss):.3f} V' if len(Vpv_mat_ss) else 'N/A'
filas_mat_eff  = f'{np.mean(Ppv_mat_ss)/Pmpp*100:.1f} %' if len(Ppv_mat_ss) else 'N/A'
mat_suffix     = '' if MATLAB_REAL else ' *'

for i, fila in enumerate(filas):
    if   i == 0: filas[i].append(f'{Pmpp} W @ {Vmpp} V')
    elif i == 1: filas[i].append('N/A')
    elif i == 2: filas[i].append(filas_mat_ppv + mat_suffix)
    elif i == 3: filas[i].append(filas_mat_eff + mat_suffix)
    elif i == 4: filas[i].append(filas_mat_vpv + mat_suffix)
    else:        filas[i].append('—')

tabla = ax_t.table(
    cellText=filas,
    colLabels=['Métrica', 'P&O', 'PSO+MPC', f'MATLAB{"" if MATLAB_REAL else " (aprox)"}'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

for (row, col), cell in tabla.get_celld().items():
    cell.set_facecolor('#1a1a2e' if row > 0 else '#2a2a5e')
    cell.set_text_props(color='white' if col != 0 else '#aaaacc')
    cell.set_edgecolor(GRID)
    if row > 0 and col == 1:
        cell.set_text_props(color=C_PO)
    if row > 0 and col == 2:
        cell.set_text_props(color=C_PSO)
    if row > 0 and col == 3:
        cell.set_text_props(color=C_MAT)

fig3.tight_layout()
plt.show()
