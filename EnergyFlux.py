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
from matplotlib.pyplot import scatter
me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2

# Still working out Units issues...


# Get Data
probe  = 1
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
trange = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
#trange = ['2015-10-16/13:06:50', '2015-10-16/13:07:10']
fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probe,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
scm_vars = scm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
# Change shape of fields bc they're confusing (will change them back eventually)

def reform(var):
	if not isinstance(var[1][0],np.ndarray):
		newvar = np.zeros(len(var[0]))
	elif isinstance(var[1][0][0],np.ndarray):
		newvar = np.zeros([len(var[0]),len(var[1][0]),len(var[1][0][0])])
	elif isinstance(var[1][0],np.ndarray):		
		newvar = np.zeros([len(var[0]),len(var[1][0])])
	for i in range(len(var[0])-1):
		newvar[i] = var[1][i]
	return newvar


# Cadence order (high -> low): edp & scm (same) -> fgm -> fpi-des -> fpi-dis
def interp_to(var_name):
	tinterpol(B_name,var_name, newname='B')
	tinterpol(E_name,var_name, newname='E')
	tinterpol(vi_name,var_name, newname='vi')
	tinterpol(ve_name,var_name, newname='ve')
	tinterpol(scm_name,var_name, newname='B_scm')
	tinterpol(Pi_name,var_name, newname='Pi')
	tinterpol(Pe_name,var_name, newname='Pe')
	tinterpol(ni_name,var_name, newname='ni')
	tinterpol(ne_name,var_name, newname='ne')

	B = 1e-9*reform(get_data('B'))
	E = 1e-3*reform(get_data('E'))
	vi = 1e3*reform(get_data('vi'))
	ve = 1e3*reform(get_data('ve'))
	B_scm = 1e-9*reform(get_data('B_scm'))
	ni = 1e6*reform(get_data('ni'))
	ne = 1e6*reform(get_data('ne'))
	Pi = 1e-9*reform(get_data('Pi'))
	Pe = 1e-9*reform(get_data('Pe'))

	ndata = len(B)
	
	return B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata

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

ion = get_data(ni_name)
elec = get_data(ne_name)
Bfld = get_data(B_name)
Efld = get_data(E_name)



# Cadence order (high -> low): edp & scm (same) -> fgm -> fpi-des -> fpi-dis


## Energy Fluxes ##

# Poynting S = ExB
# Kinetic = 0.5*(n*mv^2)*v
# Enthalpy = 0.5*v*Tr(P)  + v dot P
# Q = ?
#%%
# Poynting Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(B_name)  
S = np.zeros_like(E)
Bnew = np.zeros_like(E)
Bold = B
for i in range(ndata-1):
	Bnew[i] = B[i,:-1]
	S[i] = np.cross(E[i],Bnew[i]) / mu0


# Electron Energy Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(ve_name)  
Ke = np.zeros_like(E)
He = np.zeros_like(E)
for i in range(ndata-1):
	Ke[i] = 0.5*me*ne[i]*ve[i]*np.linalg.norm(ve[i])**2
	He[i] = 0.5*ve[i]*(Pe[0,0,0]+Pe[0,1,1]+Pe[0,2,2]) + np.dot(ve[i],Pe[i])

# Ion Energy Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(vi_name)  
Ki = np.zeros_like(E)
Hi = np.zeros_like(E)
for i in range(ndata-1):
	Ki[i] = 0.5*mi*ni[i]*vi[i]*np.linalg.norm(vi[i])**2
	Hi[i] = 0.5*vi[i]*(Pi[0,0,0]+Pi[0,1,1]+Pi[0,2,2]) + np.dot(vi[i],Pi[i])

# Heat Flux (?)

# Default units W/m^2
store_data('S', data = {'x':Bfld.times, 'y': S})
store_data('Ke', data = {'x':elec.times, 'y': Ke})
store_data('He', data = {'x':elec.times, 'y': He})
store_data('Ki', data = {'x':ion.times, 'y': Ki})
store_data('Hi', data = {'x':ion.times, 'y': Hi})

# The relative magnitudes seems fishy.  Or did I do it wrong years ago?
names = ['S','Ke','He','Ki','Hi']
options(names, 'Color', ['b','g','r'])
tplot(['S','Ke','He','Ki','Hi'])
# %%
