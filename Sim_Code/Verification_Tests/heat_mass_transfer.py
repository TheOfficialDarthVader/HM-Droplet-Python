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


class test():
    def __init__(self, time_divisor=0.0625):
        self.c = Constants()
        self.c.drop_properties()
        self.c.gas_properties()
        self.c.get_reference_conditions()
        self.c.add_drop_properties()
        self.c.add_gas_properties()
        self.c.add_properties()
        self.p2 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=1, coupled=2)
        self.p3 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=2, coupled=2)
        self.p4 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=3, coupled=2)
        self.p5 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=4, coupled=2)

        self.time_divisor = time_divisor
        self.div = self.p2.get_tau()*self.time_divisor
        self.N = 10000

    def iter_particles(self):
        last_time = 0
        for t in range(self.N):
            if (self.p2.m_d/self.p2.m_d0 > 0.001 and
               self.p2.T_d/self.p2.T_G < 0.999):
                time1 = t * self.div
                self.p2.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if (self.p3.m_d/self.p3.m_d0 > 0.001 and
               self.p3.T_d/self.p3.T_G < 0.999):
                time1 = t * self.div
                self.p3.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if (self.p4.m_d/self.p4.m_d0 > 0.001 and
               self.p4.T_d/self.p4.T_G < 0.999):
                time1 = t * self.div
                self.p4.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if (self.p5.m_d/self.p5.m_d0 > 0.001 and
               self.p5.T_d/self.p5.T_G < 0.999):
                time1 = t * self.div
                self.p5.iterate(time1 - last_time)
                last_time = time1
            else:
                break

    def plot_data(self):
        f1 = plt.figure(figsize=(20, 10))
        ax1 = f1.add_subplot(111)
        ax1.plot(self.p2.times, self.p2.diameter_2_history, 'g--',
                 label='Forward Euler')
        ax1.plot(self.p3.times, self.p3.diameter_2_history, 'rx',
                 label='Backward Euler')
        ax1.plot(self.p4.times, self.p4.diameter_2_history, 'y*',
                 label='Modified Euler')
        ax1.plot(self.p5.times, self.p5.diameter_2_history, 'x',
                 label='Runge Kutta')
        ax1.set_xlim(0)
        ax1.set_ylim(0)
        plt.xlabel(r'$t$ ($s$)')
        plt.ylabel(r'$D^2$ ($mm^2$)')
        plt.legend(loc='upper right')
        plt.title('Diameter Evolution of Evaporating Droplet')

        f2 = plt.figure(figsize=(20, 10))
        ax2 = f2.add_subplot(111)
        ax2.plot(self.p2.times, self.p2.temp_history, 'g--',
                 label='Forward Euler')
        ax2.plot(self.p3.times, self.p3.temp_history, 'rx',
                 label='Backward Euler')
        ax2.plot(self.p4.times, self.p4.temp_history, 'y*',
                 label='Modified Euler')
        ax2.plot(self.p5.times, self.p5.temp_history, 'x',
                 label='Runge Kutta')
        ax2.set_xlim(0)
        ax2.set_ylim(self.p2.T_d0)
        plt.xlabel(r'$t$ ($s$)')
        plt.ylabel(r'$T_d$ ($K$)')
        plt.legend(loc='lower right')
        plt.title('Temperature Evolution of Evaporating Droplet')

    def save_data(self):
        self.file_dir = 'Sim_Code//Verification_Tests//heat_mass_data//'
        with open(self.file_dir +
                  'c_fe_heat_mass_transfer_time_step_' + str(self.time_divisor)
                  + '_tau.txt', 'w') as f:
            self.p2.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p2.times)):
                f.write(str(self.p2.times[i]) + ' ' +
                        str(self.p2.temp_history[i]) + ' ' +
                        str(self.p2.diameter_2_history[i]) + ' ' + '\n')
        self.p2.times_temp_nd[::-1]

        with open(self.file_dir +
                  'c_be_heat_mass_transfer_time_step_' + str(self.time_divisor)
                  + '_tau.txt', 'w') as f:
            self.p3.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p3.times)):
                f.write(str(self.p3.times[i]) + ' ' +
                        str(self.p3.temp_history[i]) + ' ' +
                        str(self.p3.diameter_2_history[i]) + ' ' + '\n')
        self.p3.times_temp_nd[::-1]

        with open(self.file_dir +
                  'c_me_heat_mass_transfer_time_step_' + str(self.time_divisor)
                  + '_tau.txt', 'w') as f:
            self.p4.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p4.times)):
                f.write(str(self.p4.times[i]) + ' ' +
                        str(self.p4.temp_history[i]) + ' ' +
                        str(self.p4.diameter_2_history[i]) + ' ' + '\n')
        self.p4.times_temp_nd[::-1]

        with open(self.file_dir +
                  'c_rk_heat_mass_transfer_time_step_' + str(self.time_divisor)
                  + '_tau.txt', 'w') as f:
            self.p5.times[::-1]
            f.write('time' + ' ' + 'T_d' + ' ' + 'd2' + ' ' + '\n')
            for i in range(len(self.p5.times)):
                f.write(str(self.p5.times[i]) + ' ' +
                        str(self.p5.temp_history[i]) + ' ' +
                        str(self.p5.diameter_2_history[i]) + ' ' + '\n')
        self.p5.times[::-1]


def run_test(save=False):
    t = test()
    t.iter_particles()
    t.plot_data()
    if save is True:
        t.save_data()
