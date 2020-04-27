# -*- coding: utf-8 -*-
from Sim_Code.Objects.Particle import particle, Constants
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['pgf.texsystem'] = 'lualatex'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'font.size': 20})


class run_sim():
    def __init__(self, drop_species='decane', gas_species='air', T_G=1000,
                 rho_G=0.3529, C_p_G=1135, Re_d=17, T_d=315, D=0.004):
        self.drop_species = drop_species
        self.gas_species = gas_species
        self.T_G = T_G
        self.rho_G = rho_G
        self.C_p_G = C_p_G
        self.Re_d = Re_d
        self.T_d = T_d
        self.D = D

        self.c = Constants(self.drop_species, self.gas_species, self.T_G,
                           self.rho_G, self.C_p_G, self.Re_d)
        self.c.drop_properties()
        self.c.gas_properties()
        self.c.get_reference_conditions()
        self.c.add_drop_properties()
        self.c.add_gas_properties()
        self.c.add_properties()
        self.p = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                          D=self.D, T_d=self.T_d,
                          ODE_solver=2, coupled=2)

        self.div = self.p.get_tau()/32
        self.N = 10000

    def iter_particles(self):
        last_time = 0
        for t in range(self.N):
            if (self.p.m_d/self.p.m_d0 > 0.001 and
               self.p.T_d/self.p.T_G < 0.999):
                time1 = t * self.div
                self.p.iterate(time1 - last_time)
                last_time = time1
            else:
                break

    def plot_data(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        ax1.plot(self.p.times, self.p.diameter_2_history, '--')
        ax1.set_xlim(0)
        ax1.set_ylim(0)
        ax1.set_xlabel(r'$t$ ($s$)')
        ax1.set_ylabel(r'$D^2$ ($mm^2$)')
        ax1.set_title('Diameter Evolution of Evaporating ' +
                      self.drop_species.title() + ' Droplet')

        ax2.plot(self.p.times, self.p.temp_history, '--')
        ax2.set_xlim(0)
        ax2.set_ylim(self.p.T_d0)
        ax2.set_xlabel(r'$t$ ($s$)')
        ax2.set_ylabel(r'$T_d$ ($K$)')
        ax2.set_title('Temperature Evolution of Evaporating ' +
                      self.drop_species.title() + ' Droplet')

    def save_data(self):
        self.file_dir = 'Sim_Code//Simulation//sim_data//'
        with open(self.file_dir + 'c_' + self.drop_species +
                  '_heat_mass_transfer_time_step_tau_32.txt', 'w') as f:
            self.p.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p.times)):
                f.write(str(self.p.times[i]) + ' ' +
                        str(self.p.temp_history[i]) + ' ' +
                        str(self.p.diameter_2_history[i]) + ' ' + '\n')
        self.p.times_temp_nd[::-1]


def run_sims(save=False):
    drop_species = ['water', 'hexane', 'decane']
    T_G = [298, 437, 1000]
    rho_G = [1.184, 0.807, 0.3529]
    C_p_G = [1007, 1020, 1141]
    Re_d = [0, 110, 17]
    T_d = [282, 281, 315]
    D = [np.sqrt(1.1)/1000, 0.00176, 0.002]
    for i in range(len(drop_species)):
        r = run_sim(drop_species[i], 'air', T_G[i],
                    rho_G[i], C_p_G[i], Re_d[i], T_d[i], D[i])
        r.iter_particles()
        r.plot_data()
        if save is True:
            r.save_data()
