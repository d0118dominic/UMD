# Use pyspedas to do the multi-spacecrft calculations like divergence & curl
#   (work in progress)

#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options
from pyspedas import tinterpol
from pyspedas.mms import mec,fgm,fpi,edp,curlometer
from pytplot import get_data, store_data
from matplotlib.pyplot import plot

# Define trange and get mec data from all 4 spacecraft
probes = [1,2,3,4]
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
mec_vars = mec(probe = probes,trange=trange,data_rate='brst',time_clip=True)
fgm_vars = fgm(probe = probes, data_rate = 'brst', trange=trange,time_clip=True)
#%%
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

## This block defines all the functions necessary to get:
# Reciprocal vectors, divergence and curl for MMS and measured quantities

# Gets the separation vector pointing from a to b
def sepvec(a,b):
    sepvec = b-a
    return sepvec

# Gets a list of the reciprocal vectors [k1,k2,k3,k4]
def recip_vecs(pos1,pos2,pos3,pos4):
    
    # Define every possible separation vector
    # There has to be a cleaner way to do this part.  Keep for now
    pos12 = sepvec(pos1,pos2)
    pos13 = sepvec(pos1,pos3)
    pos14 = sepvec(pos1,pos4)
    pos23 = sepvec(pos2,pos3)
    pos24 = sepvec(pos2,pos4)
    pos34 = sepvec(pos3,pos4)
    pos21 = -pos12
    pos31 = -pos13
    pos41 = -pos14
    pos32 = -pos23
    pos42 = -pos24
    pos43 = -pos34

    # Function to calculate k vector given separation vectors a,b,c
    # where the form is k = axb/(c dot axb)
    def kvec(a,b,c):
        crossp = np.cross(a,b)
        denom = np.dot(c,crossp)
        k = crossp/denom
        return k 

    # Get each k vector and put in a list
    k1 = kvec(pos23,pos24,pos21)
    k2 = kvec(pos34,pos31,pos32)
    k3 = kvec(pos41,pos42,pos43)
    k4 = kvec(pos12,pos13,pos14)

    klist = np.array([k1,k2,k3,k4])

    return klist


# Define divergence given a veclist and klist
# where veclist is a list of some vector quantity measured at [MMS1,MMS2,MMS3,MMS4]
# and klist is the list of reciprocal vectors [k1,k2,k3,k4]

def div(veclist, klist):
    i = 0
    div = 0
    for i in range(4):
        div = div + np.dot(klist[i],veclist[i])
    return div 
def curl(veclist, klist):
    i = 0
    crl = np.array([0,0,0])
    for i in range(4):
        crl = crl + np.cross(klist[i],veclist[i])
    return crl
##%%

# Getting Div and Curl of B (or any vector field.  just change fld_names in for loop)

# Get all Bs and positions in np form and interpolted together
# [0,1,2,3] = MMS[1,2,3,4]
fld_names = [] # Names of the fields
pos_names = [] # Names of positions 
fld_interp_names = [] # interpolated version of field names 
flds = [] # get_data for each field
posits = [] # get_data for each position


# Factors to convert fld & position to SI units (change fldfact depending on fld, keep posfact as is)
fldfact = 1e-9  
posfact = 1e3 


# Get field and mec (position) data
for i in range(4):
    fld_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  
    pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
    tinterpol(fld_names[i],pos_names[i],newname = 'B' + str(i+1)) #interpolate
    fld_interp_names.append('B' + str(i+1))
    flds.append(get_data(fld_interp_names[i]))
    posits.append(get_data(pos_names[i]))

# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = flds[0].times
ndata = len(timeax)
crl = np.zeros([ndata,3])
divr = np.zeros([ndata])


# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    flds[i] = fldfact*reform(flds[i]) # [fld1,fld2,fld3,fld4]
    posits[i] = posfact*reform(posits[i]) # [pos1,pos2,pos3,pos4]


##%%
# If fld is 4D (usually because total is included), chop off the 4th term
if len(flds[0][0]) == 4:   # Just put this here so its convenient to minimize in vscode
    fld1 = np.zeros([ndata,3])
    fld2 = np.zeros([ndata,3])
    fld3 = np.zeros([ndata,3])
    fld4 = np.zeros([ndata,3])
    for i in range(ndata-1):
        fld1[i] = flds[0][i][:-1]
        fld2[i] = flds[1][i][:-1] 
        fld3[i] = flds[2][i][:-1]
        fld4[i] = flds[3][i][:-1]
    flds = [fld1,fld2,fld3,fld4]

# Get Div & Curl
for i in range(ndata-1):
    veclist = [flds[0][i],flds[1][i],flds[2][i],flds[3][i]]
    klist = recip_vecs(posits[0][i],posits[1][i],posits[2][i],posits[3][i])
    crl[i] = curl(veclist,klist)
    divr[i] = div(veclist,klist)

# If you want to convert to more convenient non-SI units, do so here.
store_data('curl', data = {'x':timeax, 'y': crl})
options('curl', 'Color', ['b','g','r'])
options('curl', 'ytitle', 'Curl(B)')

store_data('div', data = {'x':timeax, 'y': divr})
options('div', 'ytitle', 'Divergence(B)')


tplot(['curl','div'])
# %%
