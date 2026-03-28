"""
base.py — Interfaz (protocolo) para algoritmos MPPT personalizados.

Para agregar un nuevo método MPPT solo necesitas crear una clase con
los métodos definidos aquí.  No hay herencia obligatoria; basta con
que la clase tenga la firma correcta (duck typing).

══════════════════════════════════════════════════════════════════════
MODO 1 — Espacio D  (como P&O)
══════════════════════════════════════════════════════════════════════
El algoritmo devuelve directamente el Duty Cycle D [0,1].
El motor NO agrega lazo de control interno (sin MPC ni steady_state_D).

Firma requerida:
    step(V: float, I: float, D: float) → float

    Entradas
    --------
    V  : voltaje del panel medido en este instante [V]
    I  : corriente del panel medido en este instante [A]
    D  : duty cycle actual del convertidor boost [adimensional, 0..1]

    Salida
    ------
    D_nuevo : nuevo duty cycle a aplicar [adimensional, 0..1]

Opcional:
    reset(D0: float) → None   (para reiniciar el estado interno)

Ejemplo mínimo
--------------
class MiMPPT_D:
    def __init__(self):
        self._P_prev = None
        self._D_last = 0.375

    def step(self, V, I, D):
        P = V * I
        if self._P_prev is None:
            self._P_prev = P
            return D
        dP = P - self._P_prev
        self._P_prev = P
        delta = 0.004
        D_nuevo = D + (delta if dP > 0 else -delta)
        self._D_last = float(max(0.05, min(0.95, D_nuevo)))
        return self._D_last

    def reset(self, D0=0.375):
        self._P_prev = None
        self._D_last = D0

Registro en el motor
--------------------
from simulation_engine import SimulationEngine
from mi_mppt import MiMPPT_D

engine = SimulationEngine(
    mppt_algo = 'custom',
    mppt_mode = 'po',        # ← indica que devuelve D
    mppt_obj  = MiMPPT_D(),
)

══════════════════════════════════════════════════════════════════════
MODO 2 — Espacio Vref  (como INC / PSO)
══════════════════════════════════════════════════════════════════════
El algoritmo devuelve una tensión de referencia Vref [V].
El motor convierte Vref → D con MPC analítico o steady_state_D.

Firma mínima:
    step(V: float, I: float) → float

Firma extendida (opcional, con irradiancia y temperatura):
    step(V: float, I: float, G: float, T: float) → float

    Entradas
    --------
    V  : voltaje del panel [V]
    I  : corriente del panel [A]
    G  : irradiancia [W/m²]    (opcional — inspección automática)
    T  : temperatura [°C]       (opcional — inspección automática)

    Salida
    ------
    Vref : tensión de referencia a seguir [V]

Opcional:
    reset(Vref0: float) → None

Ejemplo mínimo
--------------
class MiMPPT_Vref:
    def __init__(self):
        self.Vref = 18.1
        self._V_prev = self._I_prev = None

    def step(self, V, I):
        if self._V_prev is None:
            self._V_prev, self._I_prev = V, I
            return self.Vref
        dV = V - self._V_prev
        dI = I - self._I_prev
        if abs(dV) > 1e-4:
            if dI / dV + I / V < 0:
                self.Vref += 0.5
            else:
                self.Vref -= 0.5
        self._V_prev, self._I_prev = V, I
        return float(max(0.5, min(22.2, self.Vref)))

    def reset(self, Vref0=18.1):
        self.Vref = Vref0
        self._V_prev = self._I_prev = None

Registro en el motor
--------------------
engine = SimulationEngine(
    mppt_algo = 'custom',
    mppt_mode = 'vref',      # ← indica que devuelve Vref
    mppt_obj  = MiMPPT_Vref(),
    use_mpc   = False,       # False → D analítico; True → MPC cuadrático
)

══════════════════════════════════════════════════════════════════════
Clase base opcional (no obligatoria)
══════════════════════════════════════════════════════════════════════
Si quieres aprovechar hints de tipo, puedes heredar de estas clases:
"""

import inspect


class MPPT_D_Base:
    """Clase base opcional para MPPT en espacio D."""

    def step(self, V: float, I: float, D: float) -> float:
        raise NotImplementedError

    def reset(self, D0: float = 0.375) -> None:
        pass


class MPPT_Vref_Base:
    """Clase base opcional para MPPT en espacio Vref."""

    def step(self, V: float, I: float,
             G: float = None, T: float = None) -> float:
        raise NotImplementedError

    def reset(self, Vref0: float = 18.1) -> None:
        pass


def detectar_modo(obj) -> str:
    """
    Intenta inferir el modo ('po' o 'vref') inspeccionando la firma de step().
    Devuelve 'po' si step() acepta 3+ params posicionales, 'vref' si acepta 2.
    """
    try:
        sig    = inspect.signature(obj.step)
        params = [p for p in sig.parameters.values()
                  if p.default is inspect.Parameter.empty
                  and p.name != 'self']
        return 'po' if len(params) >= 3 else 'vref'
    except Exception:
        return 'vref'
