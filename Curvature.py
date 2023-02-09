# Calculate magnetic field curvature with MMS


#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options, tplot_options
from pyspedas import tinterpol
from pyspedas.mms import mec,fgm,fpi,edp,curlometer
from pytplot import get_data, store_data
import functools
from matplotlib.pyplot import plot

me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2
eps0 = 8.85e-12   # C^2/Nm^2
#%%
# Define trange and get mec data from all 4 spacecraft
probes = [1,2,3,4]
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
mec_vars = mec(probe = probes,trange=trange,data_rate='brst',time_clip=True)
fgm_vars = fgm(probe = probes, data_rate = 'brst', trange=trange,time_clip=True)
#%%
# Reform Function to reduce tplot objects to np arrays
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


def curvature(b):
    bhat = b/np.linalg.norm(b)
    #gradb = gradient(b)
    return None


# Gets the separation vector pointing from a to b
def sepvec(a,b):
    sepvec = b-a
    return sepvec

# Gets a list of the reciprocal vectors [k1,k2,k3,k4]
def recip_vecs(pos1,pos2,pos3,pos4):
    # Define every possible separation vector
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

def grad(veclist, klist):
    i = 0
    grd = np.array([0,0,0])
    for i in range(4):
        grd = grd + klist[i]*np.transpose(veclist[i])
    return grd

#%%
# Getting  Curl of B 
# Get all Bs and positions in np form and interpolted together
# [0,1,2,3] = MMS[1,2,3,4]
B_names, pos_names,Binterp_names,Bflds,posits= [],[],[],[],[]

# Factors to convert fld & position to SI units (change fldfact depending on fld, keep posfact as is)
Bfact = 1e-9  
posfact = 1e3 
# Get field and mec (position) data
for i in range(4):
    B_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  
    pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
    tinterpol(B_names[i],pos_names[i],newname = 'B' + str(i+1)) #interpolate
    Binterp_names.append('B' + str(i+1))
    Bflds.append(get_data(Binterp_names[i]))
    posits.append(get_data(pos_names[i]))
# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = Bflds[0].times
ndata = len(timeax)
grd = np.zeros([ndata,3])

# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    Bflds[i] = Bfact*reform(Bflds[i]) # [fld1,fld2,fld3,fld4]
    posits[i] = posfact*reform(posits[i]) # [pos1,pos2,pos3,pos4]

## Get Curlometer J ##
for i in range(ndata-1):
    veclist = [Bflds[0][i][:-1],Bflds[1][i][:-1],Bflds[2][i][:-1],Bflds[3][i][:-1]]
    klist = recip_vecs(posits[0][i],posits[1][i],posits[2][i],posits[3][i])
    grd[i] = grad(veclist,klist)


# %%
B_avg, Rc = np.zeros([ndata,3]), np.zeros(ndata)
for i in range(ndata-1):
    B_avg[i] =  (Bflds[0][i][:-1]+Bflds[1][i][:-1]+Bflds[2][i][:-1]+Bflds[3][i][:-1])/4   # Is this techincally correct?  Just a 4 pt avg
    Rc[i] = (np.dot(B_avg[i],grd[i]))

# May need to revisit method or smooth fields 

# %%
