# Script to obrain various Energy fluxes
#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options
from pyspedas import tinterpol
from pyspedas.mms import mec,fgm,fpi,edp,scm
from pytplot import get_data, store_data
from matplotlib.pyplot import plot

# Get Data
probe  = 2
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probe,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
scm_vars = scm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
# Change shape of fields bc they're confusing (will change them back eventually)

def reform(var):
	if isinstance(var[1][0][0],np.ndarray):
		newvar = np.zeros([len(var[0]),len(var[1][0]),len(var[1][0][0])])
	elif isinstance(var[1][0],np.ndarray):		
		newvar = np.zeros([len(var[0]),len(var[1][0])])
	else:
		newvar = np.zeros([len(var[0])])
	for i in range(0,len(var[0])-1):
		newvar[i] = var[1][i]
	return newvar


def interp_to(var_name):
	tinterpol(B_name,var_name, newname='B')
	tinterpol(E_name,var_name, newname='E')
	tinterpol(vi_name,var_name, newname='vi')
	tinterpol(ve_name,var_name, newname='ve')
	tinterpol(scm_name,var_name, newname='B_scm')
	tinterpol(Pi_name,var_name, newname='Pi')
	tinterpol(Pe_name,var_name, newname='Pe')

	B = 1e-9*reform(get_data('B'))
	E = 1e-3*reform(get_data('E'))
	vi = 1e3*reform(get_data('vi'))
	ve = 1e3*reform(get_data('ve'))
	B_scm = 1e-9*reform(get_data('B_scm'))
	Pi = 1e-9*reform(get_data('Pi'))
	Pe = 1e-9*reform(get_data('Pe'))
	
	return B,E,vi,ve,B_scm,Pi,Pe
#%%

# Field names variables
B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
E_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
vi_name = 'mms' + str(probe) + '_' + 'dis' + '_bulkv_gse_brst'
ve_name = 'mms' + str(probe) + '_' + 'des' + '_bulkv_gse_brst'
B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
ne_name = 'mms' + str(probe) + '_' + 'des' + '_numberdensity_brst'
ni_name = 'mms' + str(probe) + '_' + 'dis' + '_numberdensity_brst'
scm_name = 'mms' + str(probe) + '_scm_acb_gse_scb_brst_l2'
Pi_name = 'mms' + str(probe) + '_dis_prestensor_gse_brst'
Pe_name = 'mms' + str(probe) + '_des_prestensor_gse_brst'


B,E,vi,ve,B_scm,Pi,Pe = interp_to(vi_name)  # Cadence order (high -> low): edp & scm (same) -> fgm -> fpi-des -> fpi-dis



#%%
# Poynting Flux
# S = ExB

# Kinetic Energy Flux
# K = 0.5*(n*mv^2)*v

# Enthalpy Flux
# H = 

# Heat Flux (?)
