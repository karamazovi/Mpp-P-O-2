# Documentación Técnica — Simulador MPPT P&O en Python

Equivalente Python del sistema MATLAB/Simulink descrito en el artículo.
Carpeta: `simulacion_python/`

---

## Índice

1. [Arquitectura general](#1-arquitectura-general)
2. [pv_panel.py — Modelo del panel fotovoltaico](#2-pv_panelpy--modelo-del-panel-fotovoltaico)
3. [boost_converter.py — Convertidor boost](#3-boost_converterpy--convertidor-boost)
4. [mppt_p_and_o.py — Algoritmo MPPT P&O](#4-mppt_p_and_opy--algoritmo-mppt-po)
5. [simulation.py — Simulación principal](#5-simulationpy--simulación-principal)
6. [simulation_animation.py — Animación interactiva](#6-simulation_animationpy--animación-interactiva)
7. [Perfil de irradiancia y validación con MATLAB](#7-perfil-de-irradiancia-y-validación-con-matlab)
8. [Dependencias](#8-dependencias)
9. [Cómo ejecutar](#9-cómo-ejecutar)

---

## 1. Arquitectura general

```
irradiancia(t)
     │
     ▼
┌──────────┐    Vpv, Ipv    ┌──────────────┐    D    ┌───────────────┐
│  PanelPV │ ─────────────► │  MPPT_PandO  │ ───────► │ BoostConverter│
└──────────┘                └──────────────┘         └───────────────┘
     ▲                                                       │
     └─────────────── Vci (= Vpv) ◄─────────────────────────┘
```

El panel entrega corriente `Ipv` al boost. El boost controla el voltaje de entrada `Vci` (= `Vpv`) mediante el duty cycle `D`. El MPPT P&O ajusta `D` en cada período de muestreo para llevar `Vpv` al punto de máxima potencia (MPP).

---

## 2. `pv_panel.py` — Modelo del panel fotovoltaico

### Modelo matemático

Se usa el **modelo de diodo simple** (single-diode model):

```
I = Iph - I0 * (exp((V + I·Rs) / (Ns·Vt)) - 1) - (V + I·Rs) / Rp
```

donde:

| Símbolo | Descripción | Valor calibrado |
|---------|-------------|----------------|
| `Ns`    | Celdas en serie | 36 |
| `Iph`   | Corriente fotogenerada | proporcional a G |
| `I0`    | Corriente de saturación inversa | 1.8826×10⁻¹⁰ A |
| `Rs`    | Resistencia serie | 0.2906 Ω |
| `Rp`    | Resistencia paralelo (shunt) | 462.04 Ω |
| `n`     | Factor de idealidad | 1.0 |
| `Vt`    | Voltaje térmico = n·k·T/q | ~0.02585 V @ 25°C |

### Parámetros de referencia del panel (artículo)

| Parámetro | Valor |
|-----------|-------|
| `Isc_ref` | 5.0 A |
| `Voc_ref` | 22.2 V |
| `Vmpp_ref`| 18.1 V |
| `Impp_ref`| 4.7 A |
| `Pmpp_ref`| 85.07 W |

### Calibración (`_calibrar`)

Los parámetros `Rs` y `Rp` se obtienen resolviendo numéricamente (con `scipy.fsolve`) un sistema de **2 ecuaciones simultáneas**:

- **Ecuación 1** — El punto MPP está sobre la curva I-V:
  ```
  Impp = Iph - I0*(exp(Vm/NsVt) - 1) - Vm/Rp
  donde Vm = Vmpp + Impp·Rs
  ```

- **Ecuación 2** — La derivada dP/dV = 0 en el MPP (condición de máximo real):
  ```
  Impp + Vmpp · (dI/dV)|_MPP = 0
  ```

> **Por qué es importante esta calibración:** Sin imponer `dP/dV=0`, el máximo de la curva P-V del modelo puede estar en un voltaje diferente a `Vmpp_ref`. Esto haría que el P&O converja a un punto incorrecto aunque pase por `(Vmpp, Impp)`.

`I0` se obtiene analíticamente a partir de la condición de circuito abierto (`I=0` en `V=Voc`):

```
I0 = (Iph - Voc/Rp) / (exp(Voc/NsVt) - 1)
```

### Tabla de lookup (LUT) (`_construir_lut`)

Para acelerar la simulación (120 000 pasos), la ecuación implícita del diodo se resuelve una sola vez al inicio usando `scipy.brentq` en 2000 puntos de voltaje, y se construye un interpolador lineal con `scipy.interp1d`.

- **Sin LUT:** ~10 minutos de simulación (120 000 llamadas a `brentq`)
- **Con LUT:** ~5 segundos (120 000 interpolaciones lineales)

La LUT se **reconstruye automáticamente** cuando cambia `G` o `T` — esto ocurre ~8 veces durante la simulación (en cada transición del perfil de irradiancia).

### Efecto de la irradiancia variable

`Iph` escala linealmente con `G`:

```python
Iph = Iph_ref * (G / 1000.0)
```

Cuando `G` baja de 1000 a 100 W/m², `Iph` cae 10×, lo que desplaza toda la curva I-V hacia abajo: `Isc` y `Pmpp` caen proporcionalmente, pero `Voc` cae solo ligeramente (escala logarítmica con I0).

---

## 3. `boost_converter.py` — Convertidor boost

### Topología

```
Vpv ──[L, RL]──┬──[MOSFET Ron]──┐
               │                 │
              [Ci]              [Co] ── [RB] ── VB (batería)
               │                 │
              GND               GND
```

### Modelo promediado (averaged model)

En lugar de simular el conmutado ciclo a ciclo, se usan las **ecuaciones promediadas** que son válidas cuando la frecuencia de conmutación es mucho mayor que la dinámica del sistema:

```
dVci/dt = (Ipv - IL) / Ci
dIL/dt  = (Vci - (1-D)·Vco - IL·(RL + D·Ron)) / L
```

### Tratamiento quasi-estático de Vco

La constante de tiempo del capacitor de salida es:

```
τ_Co = Co · RB = 22µF × 69mΩ = 1.52 µs
```

Como `τ_Co << dt = 20 µs`, la ODE de `Vco` es **numéricamente inestable** con integración Euler (el paso supera la constante de tiempo ~13×). La solución es usar el **equilibrio instantáneo**:

```python
Vco = VB + (1 - D) · IL · RB    # dVco/dt = 0
```

Esto equivale a asumir que `Vco` se estabiliza instantáneamente a su valor de estado estacionario, lo cual es válido dado que `τ_Co` es muy pequeño.

### Parámetros del convertidor

| Parámetro | Símbolo | Valor |
|-----------|---------|-------|
| Capacitor entrada | `Ci` | 22 µF |
| Capacitor salida  | `Co` | 22 µF |
| Inductancia | `L` | 330 µH |
| Resistencia inductor | `RL` | 60 mΩ |
| Resistencia MOSFET on | `Ron` | 35 mΩ |
| Resistencia batería | `RB` | 69 mΩ |
| Voltaje batería | `VB` | 24 V |
| Paso de integración | `dt` | 20 µs |

### Relación Vpv ↔ D en estado estacionario

```
Vpv = VB · (1 - D)    →    D = 1 - Vpv/VB
```

Para `Vpv = Vmpp = 18.1V` y `VB = 24V`:

```
D_ss = 1 - 18.1/24 = 0.246
```

### Constante de tiempo del inductor

```
τ_L = L / RL = 330µH / 60mΩ = 5.5 ms
```

El P&O debe esperar al menos `3·τ_L ≈ 16.5 ms` entre perturbaciones para que el inductor se estabilice. Por eso se usa `N_mppt = 1000` pasos × 20 µs = **20 ms**.

---

## 4. `mppt_p_and_o.py` — Algoritmo MPPT P&O

### Principio de operación

El algoritmo **Perturbar y Observar** funciona en dos pasos:

1. **Perturbar:** cambia `D` en ±`delta_d`
2. **Observar:** compara la potencia antes y después

### Tabla de decisión

| ΔP | ΔV | Acción | Efecto en Vpv |
|----|----|--------|---------------|
| > 0 | < 0 | D += delta_d | Vpv sigue bajando (acercando al MPP desde la derecha) |
| > 0 | > 0 | D -= delta_d | Vpv sigue subiendo (acercando al MPP desde la izquierda) |
| < 0 | > 0 | D += delta_d | Inversa: Vpv baja (alejó del MPP) |
| < 0 | < 0 | D -= delta_d | Inversa: Vpv sube (alejó del MPP) |

> **Nota de topología:** En un boost, `D↑ → Vpv↓` y `D↓ → Vpv↑`, que es la inversa de un buck. Las reglas de la tabla ya tienen esto incorporado correctamente.

### Parámetros

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `delta_d` | 0.004 | Tamaño del paso de perturbación. Más grande = convergencia más rápida pero mayor oscilación en estado estacionario |
| Zona muerta | `|ΔP| < 1e-6` | Evita perturbaciones cuando la potencia ya no cambia |
| Límite D | [0.05, 0.95] | Protege el convertidor de duty cycles extremos |

---

## 5. `simulation.py` — Simulación principal

### Condiciones de simulación

| Parámetro | Valor |
|-----------|-------|
| Temperatura | 25 °C |
| Irradiancia | Perfil variable (ver sección 7) |
| Voltaje batería `VB` | 24 V |
| Paso de tiempo `dt` | 20 µs |
| Tiempo total | 2.4 s |
| Número de pasos | 120 000 |
| Período P&O `N_mppt` | 1000 pasos = 20 ms |
| Condición inicial `Vci` | 15.0 V |
| Condición inicial `D` | 0.375 |

### Flujo principal del bucle

```python
for paso in range(num_pasos):
    t = paso * dt
    irradiancia = perfil_irradiancia(t)        # G variable
    Ipv = panel.calcular(irradiancia, T, Vci)  # modelo panel
    if paso % N_mppt == 0:
        D = mppt.actualizar(Vpv, Ipv, D)       # P&O cada 20ms
    Vci, IL, Vco = boost.simular(...)          # dinámica boost
```

### Salidas

- **`resultados.csv`** — columnas: `tiempo_s, Vpv_V, Ipv_A, Ppv_W, D, Vco_V, IL_A, G_Wm2`
- **Figura 1** — Evolución temporal de Vpv, Ipv, Ppv, D e irradiancia (5 subplots)
- **Figura 2** — Curva P-V con trayectoria del P&O + punto final + MPP teórico

---

## 6. `simulation_animation.py` — Animación interactiva

### Descripción

Mismo bucle de simulación que `simulation.py`, pero en lugar de mostrar los resultados estáticos al final, renderiza una **figura animada** con 6 paneles:

| Panel | Contenido |
|-------|-----------|
| Curva P-V | Curva estática + punto móvil + trayectoria de los últimos 40 pasos |
| Curva I-V | Curva estática + punto móvil |
| Vpv(t) | Evolución temporal construyéndose en tiempo real |
| Ppv(t) | Idem |
| D(t) | Idem |
| G(t) | Irradiancia en función del tiempo |

### Controles interactivos

| Control | Función |
|---------|---------|
| Slider **Paso** | Navega manualmente por cualquier instante de la simulación (0 a 2399) |
| Botón **▶ Play** | Reproduce la animación automáticamente |
| Botón **⏸ Pausa** | Detiene la animación en el paso actual |
| Botón **↺ Reset** | Vuelve al paso 0 |

### Barra de estado

En la parte inferior se muestra en tiempo real:
```
Paso 845/2399  |  t = 422 ms  |  Vpv = 18.05 V  |  Ppv = 84.8 W  |  D = 0.2489  |  99.7% del MPP
```

---

## 7. Perfil de irradiancia y validación con MATLAB

### Por qué se necesita irradiancia variable

El sistema MATLAB/Simulink del artículo usa un bloque **Signal 1** que aplica eventos de **sombreado parcial** al panel durante la simulación. Sin replicar este perfil, la comparación Python vs MATLAB solo es válida para el transitorio inicial de búsqueda del MPP, no para la respuesta ante perturbaciones externas.

### Perfil implementado

```python
def perfil_irradiancia(t):
    if (0.29 <= t < 0.44 or   # sombreado 1: ~150ms
        0.88 <= t < 0.99 or   # sombreado 2: ~110ms
        1.18 <= t < 1.29 or   # sombreado 3: ~110ms
        1.38 <= t < 1.54):    # sombreado 4: ~160ms
        return 100.0           # G = 100 W/m² (sombreado severo)
    return 1000.0              # G = 1000 W/m² (irradiancia nominal)
```

Los tiempos fueron extraídos visualmente de la gráfica de resultados MATLAB.

### Efecto en el panel durante el sombreado

Cuando `G` cae de 1000 → 100 W/m²:

| Variable | G=1000 | G=100 | Cambio |
|----------|--------|-------|--------|
| `Iph` | 5.0 A | 0.5 A | ÷10 |
| `Pmpp` | ~85 W | ~8.5 W | ÷10 |
| `Voc` | 22.2 V | ~19.8 V | -11% |
| `Vmpp` | 18.1 V | ~15.5 V | -14% |

La LUT se reconstruye automáticamente en cada transición de irradiancia.

### Comparación Python vs MATLAB

| Métrica | MATLAB | Python |
|---------|--------|--------|
| Vpv en estado estacionario | ~18.2 V | ~17.9 V |
| Ppv en estado estacionario | ~85 W | ~80 W |
| Convergencia inicial | ~400 ms | ~400 ms |
| Recuperación tras sombreado | ~100 ms | ~100 ms |

La diferencia de ~5W se debe a que MATLAB usa el modelo interno de Simscape para el panel BP2 (más parámetros internos), mientras que Python usa el modelo de diodo simple con 5 parámetros.

---

## 8. Dependencias

```
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
```

Instalación con el entorno virtual del proyecto:

```bash
.venv/Scripts/pip install numpy scipy matplotlib
```

---

## 9. Cómo ejecutar

### Simulación estática (gráficas y CSV)

```bash
.venv/Scripts/python simulacion_python/simulation.py
```

Genera:
- `simulacion_python/resultados.csv`
- Figura 1: evolución temporal
- Figura 2: curva P-V con trayectoria

### Animación interactiva

```bash
.venv/Scripts/python simulacion_python/simulation_animation.py
```

La simulación se pre-computa (~5 segundos) y luego abre la ventana interactiva con slider y botones Play/Pausa/Reset.
