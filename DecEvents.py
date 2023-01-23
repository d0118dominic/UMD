#%%

import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options

burch = ['2015-12-08/11:20:00','2015-12-08/11:21:00']
event1 = ['2015-12-08/11:27:00','2015-12-08/11:30:00']
event2 = ['2015-12-08/11:33:44','2015-12-08/11:34:53']
event3 = ['2015-12-08/11:39:14','2015-12-08/11:41:53']
allevents = ['2015-12-08/11:27:00','2015-12-08/11:41:53']

trange =event1
probe = 3
data_rate = 'brst'
datatype = 'dis'
edatatype = 'des'

pyspedas.mms.fgm(probe = probe, data_rate = data_rate, trange=trange,time_clip=True)
pyspedas.mms.edp(probe = probe, data_rate = data_rate, trange=trange,time_clip=True)
pyspedas.mms.fpi(probe = probe, data_rate = data_rate, 
    datatype = datatype + '-moms', trange=trange,time_clip=True)

pyspedas.mms.fpi(probe = probe, data_rate = data_rate, 
    datatype = edatatype + '-moms', trange=trange,time_clip=True)


## %%

####

from pyspedas import tinterpol
from pytplot import get_data, store_data

vi_name = 'mms' + str(probe) + '_' + datatype + '_bulkv_gse_brst'
ve_name = 'mms' + str(probe) + '_' + edatatype + '_bulkv_gse_brst'
B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
ne_name = 'mms' + str(probe) + '_' + edatatype + '_numberdensity_brst'
ni_name = 'mms' + str(probe) + '_' + datatype + '_numberdensity_brst'

tinterpol(B_name,vi_name, newname='B')
tinterpol(E_name,vi_name, newname='E')
tinterpol(ne_name,vi_name, newname='ne')
tinterpol(ve_name,vi_name, newname='ve')

B_newname = 'B'
E_newname = 'E'
ne_newname = 'ne'
ve_newname = 've'

# New variable data all interpolated to fpi ion cadence
vi = get_data(vi_name)
ni = get_data(ni_name)
B = get_data(B_newname) # Order is Bx, By, Bz, Btotal
E = get_data(E_newname)
ve = get_data(ve_newname)
ne = get_data(ne_newname)


#%%

# Calculate Ei' & J.E

ndata = len(vi.times)

# Change shape of fields bc they're confusing (will change them back eventually)
def reform(var):
	if isinstance(var[1][0],np.ndarray):   # Check if its a vector (of any dim > 1)
		newvar = np.zeros([len(var[0]),len(var[1][0])])
		for i in range(0,len(var[0])-1):
			for j in range(0,len(var[1][0])):
				newvar[i,j] = var[1][i][j]
	else:                                  # Assume anything else is a scalar
		newvar = np.zeros([len(var[0])])
		for i in range(0,len(var[0])-1):
			newvar[i] = var[1][i]
	return newvar

# Gets E+vxB in SI units
def get_eprime(E,v,B):
    Eprime = np.zeros([ndata,3])
    E = 1e-3*reform(E) # Reform each var to numpy array & convert to SI units
    B = 1e-9*reform(B)
    v= 1e3*reform(v)
    # MIGHT BE WORTH SMOOTHING THE FIELDS BEFORE CALCULATING.  TBD
    for i in range(0,ndata-1):
        Eprime[i,0] = E[i,0] + v[i,1]*B[i,2] - v[i,2]*B[i,1]
        Eprime[i,1] = E[i,1] + v[i,2]*B[i,0] - v[i,0]*B[i,2]
        Eprime[i,2] = E[i,2] + v[i,0]*B[i,1] - v[i,1]*B[i,0]
    return Eprime 

# Get J in SI units
def get_j(n,vi,ve):
    current = np.zeros([ndata,3])
    ve = 1e3*reform(ve)
    vi = 1e3*reform(vi)
    n = 1e6*reform(n)
    q = 1.6e-19 # charge unit in SI
    for i in range(0,ndata-1):
        for j in range(0,2):
            current[i,j] = q*n[i]*(vi[i,j] - ve[i,j])
    return current 

# Get scalar J.E in SI units
def get_jdote(n,vi,ve,E):
	j = get_j(n,vi,ve) # get j in SI units
	jdote = np.zeros(ndata)
	E = 1e-3*reform(E) # Reform each var to numpy array & convert to SI units
	for i in range(0,ndata-1):
		jdote[i] = j[i,0]*E[i,0] + j[i,1]*E[i,1] + j[i,2]*E[i,2]
	return jdote 



# Gets E_parallel in SI units
def get_Epar(E,B):
	Epar = np.zeros(ndata)
	E = 1e-3*reform(E) # Reform each var to numpy array & convert to SI units
	B = 1e-9*reform(B)

	######### Lazy specific fix. Make init nan values = first numerical
	for i in range(0,4): B[i] = B[4]
	##################################

	# MIGHT BE WORTH SMOOTHING THE FIELDS BEFORE CALCULATING.  TBD
	for i in range(0,ndata-1):
		Epar[i] = (E[i,0]*B[i,0] + E[i,1]*B[i,1] + E[i,2]*B[i,2])/np.linalg.norm(B[i])
	return Epar # Convert final product to mV/m


# Gets J_parallel in SI units
def get_Jpar(n,vi,ve,E,B):
	Jpar = np.zeros(ndata)
	E = 1e-3*reform(E) # Reform each var to numpy array & convert to SI units
	B = 1e-9*reform(B)
	j = get_j(n,vi,ve) # get current density as np array & SI units
	
	for i in range(0,ndata-1):
		Jpar[i] = (j[i,0]*B[i,0] + j[i,1]*B[i,1] + j[i,2]*B[i,2])/np.linalg.norm(B[i])
	return Jpar # Final product in SI units


def get_JE_Par(n,vi,ve,E,B):
	jpar = get_Jpar(ni,vi,ve,E,B)
	epar = get_Epar(E,B)
	je_par = jpar*epar
	return je_par


##%%


Eprime = get_eprime(E, vi, B)
JdotE = get_jdote(ni,vi,ve,E)
Jpar = get_Jpar(ni,vi,ve,E,B)
Epar = get_Epar(E,B)
JEpar = get_JE_Par(ni,vi,ve,E,B)

store_data('Ei_prime', data = {'x':vi.times, 'y': 1e3*Eprime})
options('Ei_prime', 'Color', ['b','g','r'])
options('Ei_prime', 'ytitle', 'Ei\' [mV/m]')
Ei_prime = 'Ei_prime'

store_data('E_par', data = {'x':vi.times, 'y': 1e3*Epar})
options('E_par', 'ytitle', 'Epar [mV/m]')
E_par = 'E_par'


store_data('J_par', data = {'x':vi.times, 'y': Jpar})
options('J_par', 'ytitle', 'Jpar [A/m^2]')
J_par = 'J_par'

store_data('JdotE', data = {'x':vi.times, 'y': 1e9*JdotE})
options('JdotE', 'ytitle', 'JdotE [nW/m^3]')
JdotE = 'JdotE'

store_data('JE_par', data = {'x':vi.times, 'y': 1e9*JEpar})
options('JE_par', 'ytitle', 'JE_par [nW/m^3]')
JE_par = 'JE_par'
##%%


plots = [B_name,E_name,vi_name,ni_name,JE_par,JdotE]

options(plots, 'legend_location', 'spedas')
options(plots, 'ylog', 0)
tplot(plots)

# %%
a = 5

