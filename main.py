# -*- coding: utf-8 -*-
from Sim_Code.Verification_Tests import uc_heat_transfer as uc_ht
from Sim_Code.Verification_Tests import uc_mass_transfer as uc_mt
from Sim_Code.Verification_Tests import heat_mass_transfer as hmt
from Sim_Code.Simulation import sim as s

vtest = True
etest = False
stest = False

# runs verification tests #
if vtest is True:
    uc_ht.run_test()
    uc_mt.run_test()
    hmt.run_test()

# gets error data for numerical methods #
if etest is True:
    uc_ht.run_error_checks()
    uc_mt.run_error_checks()

# runs simulations #
if stest is True:
    s.run_sims()
