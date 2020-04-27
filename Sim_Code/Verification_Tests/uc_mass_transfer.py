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
    def __init__(self, time_divisor=0.5):
        self.c = Constants(drop_species='water', gas_species='air', T_G=298,
                           rho_G=1.184, C_p_G=1007, Re_d=0)
        self.c.drop_properties()
        self.c.gas_properties()
        self.c.get_reference_conditions()
        self.c.add_drop_properties()
        self.c.add_gas_properties()
        self.c.add_properties()

        self.p1 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=0, coupled=0)
        self.p2 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=1, coupled=0)
        self.p3 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=2, coupled=0)
        self.p4 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=3, coupled=0)
        self.p5 = particle(self.c, [0, 0, 0], velocity=[0, 0, 0],
                           D=np.sqrt(1.1)/1000, T_d=282,
                           ODE_solver=4, coupled=0)

        self.time_divisor = time_divisor
        self.div = self.p2.get_tau()*(self.time_divisor)
        self.N = 10000

        self.p2_error = []
        self.p3_error = []
        self.p4_error = []
        self.p5_error = []

    def iter_particles(self):
        last_time = 0
        for t in range(self.N):
            if self.p1.m_d/self.p1.m_d0 > 0.001:
                time1 = t * self.div
                self.p1.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if self.p2.m_d/self.p2.m_d0 > 0.001:
                time1 = t * self.div
                self.p2.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if self.p3.m_d/self.p3.m_d0 > 0.001:
                time1 = t * self.div
                self.p3.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if self.p4.m_d/self.p4.m_d0 > 0.001:
                time1 = t * self.div
                self.p4.iterate(time1 - last_time)
                last_time = time1
            else:
                break

        last_time = 0
        for t in range(self.N):
            if self.p5.m_d/self.p5.m_d0 > 0.01:
                time1 = t * self.div
                self.p5.iterate(time1 - last_time)
                last_time = time1
            else:
                break

    def get_error_data(self):
        for i in range(len(self.p2.times_mass_nd)):
            self.p2_error.append(np.sqrt((self.p1.mass_history_nd[i] -
                                          self.p2.mass_history_nd[i])**2))
        self.p2_avg_error = sum(self.p2_error)/len(self.p2.times_mass_nd)
        print("p2:", self.p2_avg_error*100)

        for i in range(len(self.p1.times_mass_nd)):
            self.p3_error.append(np.sqrt((self.p1.mass_history_nd[i] -
                                          self.p3.mass_history_nd[i])**2))
        self.p3_avg_error = sum(self.p3_error)/len(self.p3.times_mass_nd)
        print("p3:", self.p3_avg_error*100)

        for i in range(len(self.p1.times_mass_nd)):
            self.p4_error.append(np.sqrt((self.p1.mass_history_nd[i] -
                                          self.p4.mass_history_nd[i])**2))
        self.p4_avg_error = sum(self.p4_error)/len(self.p4.times_mass_nd)
        print("p4:", self.p4_avg_error*100)

        for i in range(len(self.p5.times_mass_nd)):
            self.p5_error.append(np.sqrt((self.p1.mass_history_nd[i] -
                                          self.p5.mass_history_nd[i])**2))
        self.p5_avg_error = sum(self.p5_error)/len(self.p5.times_mass_nd)
        print("p5:", self.p5_avg_error*100, '\n')
        return(self.p2_avg_error*100, self.p3_avg_error*100,
               self.p4_avg_error*100, self.p5_avg_error*100)

    def plot_data(self):
        f = plt.figure(figsize=(20, 10))
        ax = f.add_subplot(111)
        ax.plot(self.p1.times_mass_nd, self.p1.diameter_2_history_nd, 'b-',
                label='Exact')
        ax.plot(self.p2.times_mass_nd, self.p2.diameter_2_history_nd, 'g--',
                label='Forward Euler')
        ax.plot(self.p3.times_mass_nd, self.p3.diameter_2_history_nd, 'rx',
                label='Backward Euler')
        ax.plot(self.p4.times_mass_nd, self.p4.diameter_2_history_nd, 'y*',
                label='Modified Euler')
        ax.plot(self.p5.times_mass_nd, self.p5.diameter_2_history_nd, 'x',
                label='Runge Kutta')
        ax.set_xlim(0)
        ax.set_ylim(0, 1)
        plt.xlabel(r'$t/\tau_{d,0}$')
        plt.ylabel(r'$D^2/{D_0}^2$')
        plt.legend(loc='upper right')
        plt.title('Non-Dimensionalised Diameter Evolution of ' +
                  'Evaporating Droplet')

    def save_data(self):
        self.file_dir = 'Sim_Code//Verification_Tests//uc_mass_data//'
        with open(self.file_dir + 'uc_an_d2_transfer_time_step_tau_' +
                  str(self.time_divisor) + '_tau_nd.txt', 'w') as f:
            self.p1.times_temp_nd[::-1]
            f.write('time' + ' ' + 'D2' + ' ' + '\n')
            for i in range(len(self.p1.times_mass_nd)):
                f.write(str(self.p1.times_mass_nd[i]) + ' ' +
                        str(self.p1.diameter_2_history_nd[i]) + '\n')
        self.p1.times_temp_nd[::-1]

        with open(self.file_dir + 'uc_fe_d2_transfer_time_step_tau_' +
                  str(self.time_divisor) + '_tau_nd.txt', 'w') as f:
            self.p2.times_temp_nd[::-1]
            f.write('time' + ' ' + 'D2' + ' ' + '\n')
            for i in range(len(self.p2.times_mass_nd)):
                f.write(str(self.p2.times_mass_nd[i]) + ' ' +
                        str(self.p2.diameter_2_history_nd[i]) + '\n')
        self.p2.times_temp_nd[::-1]

        with open(self.file_dir + 'uc_be_d2_transfer_time_step_tau_' +
                  str(self.time_divisor) + '_tau_nd.txt', 'w') as f:
            self.p3.times_temp_nd[::-1]
            f.write('time' + ' ' + 'D2' + ' ' + '\n')
            for i in range(len(self.p3.times_mass_nd)):
                f.write(str(self.p3.times_mass_nd[i]) + ' ' +
                        str(self.p3.diameter_2_history_nd[i]) + '\n')
        self.p3.times_temp_nd[::-1]

        with open(self.file_dir + 'uc_me_d2_transfer_time_step_tau_' +
                  str(self.time_divisor) + '_tau_nd.txt', 'w') as f:
            self.p4.times_temp_nd[::-1]
            f.write('time' + ' ' + 'D2' + ' ' + '\n')
            for i in range(len(self.p4.times_mass_nd)):
                f.write(str(self.p4.times_mass_nd[i]) + ' ' +
                        str(self.p4.diameter_2_history_nd[i]) + '\n')
        self.p4.times_temp_nd[::-1]

        with open(self.file_dir + 'uc_rk_d2_transfer_time_step_tau_' +
                  str(self.time_divisor) + '_tau_nd.txt', 'w') as f:
            self.p5.times_temp_nd[::-1]
            f.write('time' + ' ' + 'D2' + ' ' + '\n')
            for i in range(len(self.p5.times_mass_nd)):
                f.write(str(self.p5.times_mass_nd[i]) + ' ' +
                        str(self.p5.diameter_2_history_nd[i]) + '\n')
        self.p5.times_temp_nd[::-1]


class error_tests():
    def __init__(self):
        self.error_data = []
        self.timestep_sizes = np.arange(0.1, 1.1, 0.1)
        self.timestep_sizes = np.insert(self.timestep_sizes, 0, 0.001)

    def get_errors(self):
        for i in range(len(self.timestep_sizes)):
            self.t = test(self.timestep_sizes[i])
            self.t.iter_particles()
            self.error_data.append(self.t.get_error_data())

    def plot_errors(self):
        f = plt.figure(figsize=(20, 10))
        ax = f.add_subplot(111)
        ax.plot(self.timestep_sizes, [self.error_data[i][0] for i in
                                      range(len(self.error_data))],
                'gx', label='Forward Euler')
        ax.plot(self.timestep_sizes, [self.error_data[i][1] for i in
                                      range(len(self.error_data))],
                'rx', label='Backward Euler')
        ax.plot(self.timestep_sizes, [self.error_data[i][2] for i in
                                      range(len(self.error_data))],
                'yx', label='Modified Euler')
        ax.plot(self.timestep_sizes, [self.error_data[i][3] for i in
                                      range(len(self.error_data))],
                'bx', label='Runge Kutta')
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.xlabel(r'$\Delta t/\tau_{d}$')
        plt.ylabel('Average Percentage Error')
        plt.legend(loc='upper left')
        plt.title('Average Percentage Error For Uncoupled Mass Transfer ' +
                  'for Different Sized Timesteps')

    def save_errors(self):
        self.file_dir = 'Sim_Code//Verification_Tests//uc_mass_data//'
        with open(self.file_dir +
                  'uc_fe_mass_transfer_convergence.txt', 'w') as f:
            self.timestep_sizes[::-1]
            f.write('timestep' + ' ' + 'error' + ' ' + '\n')
            for i in range(len(self.error_data)):
                f.write(str(self.timestep_sizes[i]) + ' ' +
                        str(self.error_data[i][0]) + '\n')
        self.timestep_sizes[::-1]

        with open(self.file_dir +
                  'uc_be_mass_transfer_convergence.txt', 'w') as f:
            self.timestep_sizes[::-1]
            f.write('timestep' + ' ' + 'error' + ' ' + '\n')
            for i in range(len(self.error_data)):
                f.write(str(self.timestep_sizes[i]) + ' ' +
                        str(self.error_data[i][1]) + '\n')
        self.timestep_sizes[::-1]

        with open(self.file_dir +
                  'uc_me_mass_transfer_convergence.txt', 'w') as f:
            self.timestep_sizes[::-1]
            f.write('timestep' + ' ' + 'error' + ' ' + '\n')
            for i in range(len(self.error_data)):
                f.write(str(self.timestep_sizes[i]) + ' ' +
                        str(self.error_data[i][2]) + '\n')
        self.timestep_sizes[::-1]

        with open(self.file_dir +
                  'uc_rk_mass_transfer_convergence.txt', 'w') as f:
            self.timestep_sizes[::-1]
            f.write('timestep' + ' ' + 'error' + ' ' + '\n')
            for i in range(len(self.error_data)):
                f.write(str(self.timestep_sizes[i]) + ' ' +
                        str(self.error_data[i][3]) + '\n')
        self.timestep_sizes[::-1]


def run_test(save=False):
    t = test()
    t.iter_particles()
    t.plot_data()
    if save is True:
        t.save_data()


def run_error_checks(save=False):
    t = error_tests()
    t.get_errors()
    t.plot_errors()
    if save is True:
        t.save_errors()
