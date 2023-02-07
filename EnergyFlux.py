# Script to obrain various Energy fluxes
#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options
from pyspedas import tinterpol
from pytplot import tplot_options
from pyspedas.mms import mec,fgm,fpi,edp,scm
from pytplot import get_data, store_data
from matplotlib.pyplot import plot
from matplotlib.pyplot import scatter
me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2
eps0 = 8.85e-12   # C^2/Nm^2

# This code now works.  Produces same result as Eastwood 2020 (aside from small frame shift.)
# Everything is in SI units.  Any other unit conversion should happen at the end when making the tplot variable
##
#
# Get Data
#trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
trange = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
#trange = ['2015-10-16/13:06:50', '2015-10-16/13:07:10']

burch = ['2015-12-08/11:20:00','2015-12-08/11:21:00']
event1 = ['2015-12-08/11:27:00','2015-12-08/11:30:00']
event2 = ['2015-12-08/11:33:44','2015-12-08/11:34:53']
event3 = ['2015-12-08/11:39:14','2015-12-08/11:41:53']
event3s = ['2015-12-08/11:39:30','2015-12-08/11:41:30']
allevents = ['2015-12-08/11:27:00','2015-12-08/11:41:53']
gen = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
#

probe  = 1 
trange = trange

fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probe,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
scm_vars = scm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
#%%

# Function to change shape of fields to np arrays (will change them back eventually)
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

# LMN Conversion function
def convert_lmn(gse, lmn):
    vec = np.zeros_like(gse)
    for i in range(3):
        vec[i] = np.dot(lmn[i],gse)
    return vec


# Functions for energy densites and fluxes #
# Still unsure, but should be correct within ord of mag
# Poynting S = ExB
# Kinetic = 0.5*(n*mv^2)*v
# Enthalpy = 0.5*v*Tr(P)  + v dot P
# Q = ?
def therm_dens(P):
	ut = 0.5*np.trace(P)
	return ut

def kinetic_dens(m,n,v):
	uk = 0.5*m*n*np.linalg.norm(v)**2
	return uk

def B_dens(B):
	um = 0.5*(np.linalg.norm(B)**2)*mu0**-1
	return um

def E_dens(E):
	ue = 0.5*eps0*np.linalg.norm(E)**2
	return E

def kin_flux(m,n,v):
	K = 0.5*m*n*v**3
	#K = 0.5*m*n*v*np.linalg.norm(v)**2# this uses |v|*v
	return K

# This works now even though its literally identical to the old version that didnt work..
def enth_flux(v,P):
	H = 0.5*v*np.trace(P) + np.dot(v,P)
	return H

# Functions for Power densities (J.E, J.E', -PdotGrad(u), du/dt, Div(S))
def get_j(n,vi,ve):
	q = 1.6e-19 # charge unit in SI
	j = q*n*(vi - ve)
	return j
	
#def get_curlometer_j()

def get_Eprime(E,v,B):
	Ep = E + np.cross(v,B)
	return Ep


 

#%%
def Poynt_flux(E,B):
	S = np.cross(E,B)/mu0
	return S

# Function to read data, interpolate, convert to SI units, and come out in the form of np arrays
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
E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
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




#%%


lmn_0810 = np.array([
[0.985, -0.141, 0.097],
[0.152, 0.982, -0.109],
[-0.080, 0.122, 0.989]])

lmn_1016 = np.array([
[0.3665, -0.1201, 0.9226],
[0.5694, -0.7553, -0.3245],
[0.7358, 0.6443, -0.2084]])

lmn_0711 = np.array([
[0.94822, -0.25506, -0.18926],
[0.18182, 0.92451, -0.334996],
[0.26042, 0.28324, 0.92301]])


lmn_0617 = np.array([
[0.93, 0.3, -0.2],
[-0.27, 0.2, -0.94],
[-0.24, 0.93, 0.27]])

I = np.identity(3)

frame = I

# Poynting Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(B_name)  
S = np.zeros_like(E)
for i in range(ndata-1):
	S[i] = convert_lmn(Poynt_flux(E[i],B[i,:-1]),frame)


# Electron Energy Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(ve_name)  
Ke = np.zeros_like(E)
He = np.zeros_like(E)
for i in range(ndata-1):
	He[i] = convert_lmn(enth_flux(ve[i],Pe[i]),frame) #for some reason this works now
	#He[i] = convert_lmn(0.5*ve[i]*np.trace(Pe[i]) + np.dot(ve[i],Pe[i]),frame)
	Ke[i] = convert_lmn(kin_flux(me,ne[i],ve[i]),frame)  # this uses v^3  #unclear which is more correct. Eastwood uses v^3

# Ion Energy Flux
B,E,vi,ve,B_scm,ni,ne,Pi,Pe,ndata = interp_to(vi_name)  
Ki = np.zeros_like(E)
Hi = np.zeros_like(E)
for i in range(ndata-1):
	Hi[i] = convert_lmn(enth_flux(vi[i],Pi[i]),frame)
	#Hi[i] = convert_lmn(0.5*vi[i]*np.trace(Pi[i]) + np.dot(vi[i],Pi[i]),frame)
	Ki[i] = convert_lmn(kin_flux(mi,ni[i],vi[i]),frame)  

# Default units W/m^2
store_data('S', data = {'x':Bfld.times, 'y': S})
store_data('Ke', data = {'x':elec.times, 'y': Ke})
store_data('He', data = {'x':elec.times, 'y': He})
store_data('Ki', data = {'x':ion.times, 'y': Ki})
store_data('Hi', data = {'x':ion.times, 'y': Hi})


names = ['S','Ke','He','Ki','Hi']
options(names, 'Color', ['b','g','r'])
tplot_options('vertical_spacing',0.3)
tplot(['S','Ke','He','Ki','Hi'])
# %%
#Testtextt
# %%
