class BoostConverter:
    """
    Modelo promediado (averaged model) de un convertidor boost con batería en la salida.

    Estados:
        Vci : voltaje en el capacitor de entrada (V)
        IL  : corriente en el inductor (A)
        Vco : voltaje en el capacitor de salida (V)

    Ecuaciones de estado promediadas:
        dVci/dt = (Ipv - IL) / Ci
        dIL/dt  = (Vci - (1-D)*Vco - IL*(RL + D*Ron)) / L
        dVco/dt = ((1-D)*IL - (Vco - VB)/RB) / Co

    Integración: Euler hacia adelante con paso dt.
    """

    def __init__(self, parametros=None):
        if parametros is None:
            parametros = {}
        p = parametros

        self.Ci  = p.get('Ci',  22e-6)    # Capacitor entrada (F)
        self.Co  = p.get('Co',  22e-6)    # Capacitor salida  (F)
        self.L   = p.get('L',   330e-6)   # Inductancia       (H)
        self.RL  = p.get('RL',  60e-3)    # Resistencia inductor (Ω)
        self.Ron = p.get('Ron', 35e-3)    # Resistencia MOSFET on (Ω)
        self.RB  = p.get('RB',  69e-3)    # Resistencia batería   (Ω)
        self.dt  = p.get('dt',  2e-5)     # Paso de integración   (s)

    def simular(self, Vci, IL, Vco, VB, Ipv, D):
        """
        Avanza un paso de tiempo dt.

        Parámetros:
            Vci  : voltaje capacitor entrada (V)
            IL   : corriente inductor (A)
            Vco  : voltaje capacitor salida (V)
            VB   : voltaje batería (V)
            Ipv  : corriente del panel PV (A)
            D    : ciclo de trabajo (0 a 1)

        Retorna:
            Vci_new, IL_new, Vco_new
        """
        # Vco quasi-estático: τ = Co*RB = 22µF*69mΩ = 1.52µs << dt = 20µs
        # La ODE de Vco es numéricamente inestable con Euler (dt >> τ),
        # así que se usa el equilibrio instantáneo: dVco/dt = 0
        Vco = VB + (1.0 - D) * IL * self.RB

        # Derivadas (modelo promediado)
        dVci = (Ipv - IL) / self.Ci
        dIL  = (Vci - (1.0 - D) * Vco - IL * (self.RL + D * self.Ron)) / self.L

        # Integración Euler
        Vci_new = Vci + dVci * self.dt
        IL_new  = max(IL  + dIL  * self.dt, 0.0)   # IL no puede ser negativa
        Vco_new = Vco  # ya es quasi-estático

        return Vci_new, IL_new, Vco_new
