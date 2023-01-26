
# Script to obrain various Energy fluxes
#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options
from pytplot import tplot_options
from pyspedas import tinterpol
from pyspedas.mms import mec,fgm,fpi,edp,scm
from pytplot import get_data, store_data
from matplotlib.pyplot import plot
from matplotlib.pyplot import scatter
me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2

# Still working out Units issues...
event1 = ['2015-12-08/11:27:00','2015-12-08/11:30:00']
event2 = ['2015-12-08/11:33:44','2015-12-08/11:34:53']
event3 = ['2015-12-08/11:39:14','2015-12-08/11:41:53']

# Get Data
probe  = 1
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
trange = event1
fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
# Change shape of fields bc they're confusing (will change them back eventually)

#%%

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

def interp_to(var_name):
    tinterpol(B_name,var_name, newname='B')
    tinterpol(Pi_name,var_name, newname='Pi')
    tinterpol(Pe_name,var_name, newname='Pe')

    B = 1e-9*reform(get_data('B'))
    Pi = 1e-9*reform(get_data('Pi'))
    Pe = 1e-9*reform(get_data('Pe'))

    ndata = len(B)

    return B,Pi,Pe,ndata


def ParPerp_pres(p,B):
    num = p[0,0]*B[0]**2 + p[1,1]*B[1]**2 + p[2,2]*B[2]**2 
    + 2*p[0,1]*B[0],B[1] 
    + 2*p[0,2]*B[0],B[2] 
    + 2*p[1,2]*B[1],B[2]
    
    par = num/(np.linalg.norm(B)**2)
    perp = (np.trace(p) - par)/2

    return par,perp



def get_Q(p,B):
    par,perp = ParPerp_pres(p,B)
    I1 = np.trace(p)
    I2 = p[0,0]*(p[1,1] + p[2,2]) - (p[0,1]**2 + p[0,2]**2 + p[1,2]**2) + p[1,1]*p[2,2]
    num = 4*I2
    dem = (I1 - par)*(I1 + 3*par)
    Q = 1 - num/dem

    return Q


B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
Pe_name = 'mms' + str(probe) + '_des_prestensor_gse_brst'
Pi_name = 'mms' + str(probe) + '_dis_prestensor_gse_brst'


ion = get_data(Pi_name)
elec = get_data(Pe_name)



B,Pi,Pe,ndata = interp_to(Pi_name)
Qe = np.zeros(ndata)
Qi = np.zeros(ndata)
Q = np.zeros([ndata,2])

for i in range(ndata-1):
    Qe[i] = get_Q(Pe[i],B[i,:-1])
    Qi[i] = get_Q(Pi[i],B[i,:-1])
    Q[i] =  np.array([Qe[i],Qi[i]])

#B,Pi,Pe,ndata = interp_to(Pi_name)
#Qi = np.zeros(ndata)
#for i in range(ndata-1):
#	Qi[i] = get_Q(Pi[i],B[i])

store_data('Q', data = {'x':ion.times, 'y':np.sqrt(Q)})
options('Q', 'Color', ['b','r'])
options('Q', 'thick', 2)
options('Q','legend_names', ['Qe^1/2','Qi^1/2'])
tplot_options('var_label',['Qe','Qi'])

tplot('Q')
# %%
