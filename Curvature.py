# Calculate magnetic field curvature with MMS


#%%
import pyspedas
from pytplot import tplot
import numpy as np
from pytplot import options, tplot_options
from pyspedas import tinterpol
from pyspedas.mms import mec,fgm,fpi,edp,curlometer
from pytplot import get_data, store_data,timespan
from pyspedas.analysis.tsmooth import tsmooth
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
# trange = ['2017-06-17/20:23:50', '2017-06-17/20:24:10']

# trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']

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

# This stuff still needs to be figured out
# Still some issues with the units that come out
#/////////////////////////////////////////////////////////////
def grad(veclist, klist):
    i = 0
    mlist = []  # We need to turn these regular arrays into matrix objects
    grd = np.zeros([3,3])
    for i in range(4):
        mlist.append(np.array([veclist[i]]))
        grd = grd + klist[i]*mlist[i].T #Fix this too!!
    return grd


# The resulting units still don't seem right
def curv(veclist, klist):      # Modified version of the gradient calculation to get the curvature (1/Rc)
    i = 0
    mlist = []  # We need to turn these regular arrays into matrix objects
    vec = np.zeros(3)
    for i in range(4):
        mlist.append(np.array([veclist[i]]))
        # vec = vec + klist[i]*(veclist[i]**2)/np.linalg.norm(veclist[i])**2
        
        # vec = vec + np.dot(mlist[i],klist[i]*mlist[i].T)/(np.linalg.norm(mlist[i])**2) # This version takes the 
                # This version is the most precise with definitions (where k*b is a tensor, and the dot product is a vector) 
                # ... But the result is not very believable.  No discernable signal and tens of orders of magnitude off


        vec = vec + mlist[i]*klist[i]*mlist[i].T/(np.linalg.norm(mlist[i])**2)   
        #       # This version gets closest to believable units .. down to ~ 50km for 
        

    crv = np.linalg.norm(vec)    
    return crv
#/////////////////////////////////////////////////////////////

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
timeax = Bflds[1].times
ndata = len(timeax)

# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    Bflds[i] = Bfact*reform(Bflds[i]) # [fld1,fld2,fld3,fld4]
    posits[i] = posfact*reform(posits[i]) # [pos1,pos2,pos3,pos4]

    for j in range(ndata-1):
        if np.any(np.isnan(Bflds[i][j])):
            Bflds[i][j] = np.array([1e-15,1e-15,1e-15,1e-15])

# Getting Curvature
B_avg,grd,crv, Rc = np.zeros([ndata,3]), np.zeros([ndata,3,3]), np.zeros(ndata), np.zeros(ndata)
for i in range(ndata-1):
    veclist = [Bflds[0][i][:-1],Bflds[1][i][:-1],Bflds[2][i][:-1],Bflds[3][i][:-1]]
    # veclist = [\
    #     Bflds[0][i][:-1]/np.linalg.norm(Bflds[0][i][:-1]),\
    #     Bflds[1][i][:-1]/np.linalg.norm(Bflds[1][i][:-1]),\
    #     Bflds[2][i][:-1]/np.linalg.norm(Bflds[2][i][:-1]),\
    #     Bflds[3][i][:-1]/np.linalg.norm(Bflds[3][i][:-1])]
    B_avg[i] =  (Bflds[0][i][:-1]+Bflds[1][i][:-1]+Bflds[2][i][:-1]+Bflds[3][i][:-1])/4   # Is this techincally correct?  Just a 4 pt avg
    klist = recip_vecs(posits[0][i],posits[1][i],posits[2][i],posits[3][i])
    grd[i] = grad(veclist,klist)
    crv[i] = curv(veclist,klist)
    Rc[i] = 1/crv[i]
    #Rc[i] = np.abs((np.dot(B_avg[i],grd[i]))**-1) / np.linalg.norm(B_avg)**2

#%%


store_data('curvature', data = {'x':timeax, 'y': Rc})
tsmooth('curvature', width=0.05*len(Rc),new_names = '1/R', preserve_nans=0)
options('curvature', 'thick',1.5)
#options('smoothed', 'thick',1.5)
#options('smoothed', 'yrange',[0,0.2e26])

#timespan('2017-06-17 20:23:50', 60, keyword='seconds')
tplot('curvature')
# %%
# Next: Get gyroradii


fpi_vars = fpi(probe = probes,data_rate = 'brst',trange=trange,time_clip=True)


def interp_to(var_name):
	tinterpol(Ti_name,var_name, newname='Pi')
	tinterpol(Te_name,var_name, newname='Pe')
	Pi = 1e-9*reform(get_data('Pi'))
	Pe = 1e-9*reform(get_data('Pe'))

	ndata = len(B)
	
	return Pi,Pe,ndata

for i in range(4):
    Te_names.append('mms' + str(probes[i]) + '_des_temptensor_gse_brst') #Change this if you want to calc for another vec quantity  
    Ti_names.append('mms' + str(probes[i]) + '_dis_temptensor_gse_brst') #Change this if you want to calc for another vec quantity  
    tinterpol(B_names[i],pos_names[i],newname = 'B' + str(i+1)) #interpolate
    Binterp_names.append('B' + str(i+1))
    posits.append(get_data(pos_names[i]))
# %%
