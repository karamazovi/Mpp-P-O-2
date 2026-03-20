class MPPT_PandO:
    """
    Algoritmo Perturbar y Observar (Perturb & Observe) para MPPT.

    En cada llamada a actualizar() se compara la potencia actual con la anterior:
    - Si la potencia aumentó y el voltaje subió → seguir subiendo D
    - Si la potencia aumentó y el voltaje bajó → seguir bajando D
    - Si la potencia bajó  y el voltaje subió → invertir (bajar D)
    - Si la potencia bajó  y el voltaje bajó → invertir (subir D)

    El duty cycle D controla el boost: Vpv = Vco * (1 - D),
    por lo que aumentar D reduce Vpv y viceversa.
    """

    def __init__(self, parametros=None):
        if parametros is None:
            parametros = {}
        self.delta_d = parametros.get('delta_d', 0.004)
        self.prev_power   = None
        self.prev_voltage = None

    def actualizar(self, Vpv, Ipv, D):
        """
        Actualiza el duty cycle D según el algoritmo P&O.

        Parámetros:
            Vpv : voltaje del panel (V)
            Ipv : corriente del panel (A)
            D   : duty cycle actual

        Retorna:
            D   : duty cycle actualizado
        """
        power = Vpv * Ipv

        if self.prev_power is None:
            self.prev_power   = power
            self.prev_voltage = Vpv
            return D

        delta_p = power - self.prev_power
        delta_v = Vpv - self.prev_voltage

        if abs(delta_p) < 1e-6:
            # Potencia estable, no perturbar
            pass
        elif delta_p > 0:
            # La potencia aumentó: continuar en la misma dirección
            D += self.delta_d if delta_v < 0 else -self.delta_d
        else:
            # La potencia bajó: invertir dirección
            D += self.delta_d if delta_v > 0 else -self.delta_d

        self.prev_power   = power
        self.prev_voltage = Vpv

        # Limitar D a rango seguro
        D = max(0.05, min(0.95, D))
        return D
