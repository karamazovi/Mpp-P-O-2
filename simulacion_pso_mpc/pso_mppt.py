"""
Algoritmo PSO (Particle Swarm Optimization) para MPPT.

El PSO busca el voltaje Vref que maximiza la potencia del panel
escaneando la curva P-V con un enjambre de partículas.

Parámetros PSO (artículo):
  n_particles = 10
  w  = 0.5   (inercia)
  c1 = 1.5   (componente cognitiva — atracción al mejor personal)
  c2 = 1.5   (componente social   — atracción al mejor global)
  n_iter = 20 iteraciones por llamada
"""

import numpy as np


class PSO_MPPT:
    def __init__(self, parametros=None):
        p = parametros or {}
        self.n_particles = p.get('n_particles', 10)
        self.w           = p.get('w',           0.5)
        self.c1          = p.get('c1',          1.5)
        self.c2          = p.get('c2',          1.5)
        self.n_iter      = p.get('n_iter',      20)
        self.Vmin        = p.get('Vmin',        0.5)
        self.Vmax        = p.get('Vmax',        22.2)

        # Inicializar enjambre distribuido uniformemente en [Vmin, Vmax]
        self.pos       = np.linspace(self.Vmin, self.Vmax, self.n_particles)
        self.vel       = np.zeros(self.n_particles)
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full(self.n_particles, -np.inf)
        self.gbest_pos = self.pos[self.n_particles // 2]
        self.gbest_val = -np.inf

    def buscar_mpp(self, panel, irradiancia, temperatura):
        """
        Ejecuta n_iter iteraciones del PSO y retorna Vref (voltaje MPP).

        Parámetros:
            panel       : instancia de PanelPV
            irradiancia : W/m²
            temperatura : °C

        Retorna:
            Vref (V) — voltaje de máxima potencia encontrado
        """
        for _ in range(self.n_iter):
            # Evaluar fitness de cada partícula
            for i in range(self.n_particles):
                V = float(np.clip(self.pos[i], self.Vmin, self.Vmax))
                I = panel.calcular(irradiancia, temperatura, V)
                P = V * I

                if P > self.pbest_val[i]:
                    self.pbest_val[i] = P
                    self.pbest_pos[i] = V

                if P > self.gbest_val:
                    self.gbest_val = P
                    self.gbest_pos = V

            # Actualizar velocidades y posiciones
            r1 = np.random.uniform(0, 1, self.n_particles)
            r2 = np.random.uniform(0, 1, self.n_particles)
            self.vel = (self.w  * self.vel
                      + self.c1 * r1 * (self.pbest_pos - self.pos)
                      + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.pos = np.clip(self.pos + self.vel, self.Vmin, self.Vmax)

        return float(self.gbest_pos)

    def reiniciar(self, Vmin=None, Vmax=None):
        """Reinicia el enjambre (útil tras cambios bruscos de irradiancia)."""
        if Vmin is not None:
            self.Vmin = Vmin
        if Vmax is not None:
            self.Vmax = Vmax
        self.pos       = np.linspace(self.Vmin, self.Vmax, self.n_particles)
        self.vel       = np.zeros(self.n_particles)
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full(self.n_particles, -np.inf)
        self.gbest_val = -np.inf
