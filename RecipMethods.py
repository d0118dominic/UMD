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

# Define trange and get mec data from all 4 spacecraft
probes = [1,2,3,4]
trange = ['2016-12-09/09:02', '2016-12-09/09:04']
mec_vars = mec(probe = probes,trange=trange,data_rate='brst',time_clip=True)
fgm_vars = fgm(probe = probes, data_rate = 'brst', trange=trange,time_clip=True)

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

# %%

## This block defines all the functions necessary to get:
# Reciprocal vectors, divergence and curl for MMS and measured quantities

# Gets the separation vector pointing from a to b
def sepvec(a,b):
    sepvec = b-a
    return sepvec

# Gets a list of the reciprocal vectors [k1,k2,k3,k4]
def recip_vecs(pos1,pos2,pos3,pos4):
    positions = [pos1,pos2,pos3,pos4,pos1]
    
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

    klist = [k1,k2,k3,k4]

    return klist


# Define divergence given a veclist and klist
# where veclist is a list of some vector quantity measured at [MMS1,MMS2,MMS3,MMS4]
# and klist is the list of reciprocal vectors [k1,k2,k3,k4]
def div(veclist, klist):
    i = 0
    div = 0
    while i < 4:
        div += np.dot(klist[i],veclist[i])
        i+=1
    return div

def curl(veclist, klist):
    i = 0
    curl = np.array([0,0,0])
    while i < 4:
        curl += np.curl(klist[i],veclist[i])
        i+=1
    return curl
# %%
# Let's try getting curl(B)


##%%
# 

# Get all Bs and positions in np form and interpolted together

B1_name = 'mms' + str(probes[0]) + '_fgm_b_gse_brst_l2'
B2_name = 'mms' + str(probes[1]) + '_fgm_b_gse_brst_l2'
B3_name = 'mms' + str(probes[2]) + '_fgm_b_gse_brst_l2'
B4_name = 'mms' + str(probes[3]) + '_fgm_b_gse_brst_l2'

pos1_name = 'mms'+ str(probes[0]) + '_mec_r_gsm'
pos2_name = 'mms'+ str(probes[1]) + '_mec_r_gsm'
pos3_name = 'mms'+ str(probes[2]) + '_mec_r_gsm'
pos4_name = 'mms'+ str(probes[3]) + '_mec_r_gsm'

tinterpol(B1_name,pos1_name, newname='B1')
tinterpol(B2_name,pos2_name, newname='B2')
tinterpol(B3_name,pos3_name, newname='B3')
tinterpol(B4_name,pos4_name, newname='B4')

B1_name,B2_name,B3_name,B4_name = 'B1','B2','B3','B4'
	
B1 = get_data(B1_name)
B2 = get_data(B2_name)
B3 = get_data(B3_name)
B4 = get_data(B4_name)

pos1 = get_data(pos1_name)
pos2 = get_data(pos2_name)
pos3 = get_data(pos3_name)
pos4 = get_data(pos4_name)

ndata = len(B1.times)

B1,B2,B3,B4 = 1e-9*reform(B1), 1e-9*reform(B2), 1e-9*reform(B3),1e-9*reform(B4)
pos1,pos2,pos3,pos4 = 1e3*reform(pos1), 1e3*reform(pos2), 1e3*reform(pos3),1e3*reform(pos4)


# %%

# Next: Get reciprocal vectors, curl of B at every data point (4001 of them in this case)
