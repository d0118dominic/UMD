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
from matplotlib.pyplot import scatter

import functools
from matplotlib.pyplot import plot

me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2
eps0 = 8.85e-12   # C^2/Nm^2
vc = 3e8 #m/s
e = 1.602e-19  #C
kb = 1.380649e-23
#%%
# Define trange and get mec data from all 4 spacecraft
probes = [1,2,3,4]


# Events with easily accessible mec data
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']  # Classic.  Sep ~ 1 de
# trange = ['2016-12-09/09:03:30', '2016-12-09/09:04:30']  # Electron. Sep ~ 5 de
trange = ['2016-11-09/13:39:00', '2016-11-09/13:40:00']  # Shock.  Sep ~ 16 de
# trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00'] # Classic. Sep ~ 2 de
#trange = ['2016-10-22/12:59:00', '2016-10-22/12:59:30']

# Events without easily accesible mec data
#   there is somewhere tho, or one could input manually 
# trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
#trange = ['2017-06-17/20:23:50', '2017-06-17/20:24:10']
# trange = ['2015-10-16/13:06:50', '2015-10-16/13:07:10'] # 

# trange = ['2015-12-06/23:38:00', '2015-12-06/23:39:00']
# timespan('2016-10-22 12:59:00', 30,keyword='seconds')
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



# def gyrorad(B,T,m):
#     R = np.sqrt() 
# Not really sure if anything wrong here, just haven't checked in a while. TBD
#/////////////////////////////////////////////////////////////
def grad(veclist, klist):
    i = 0
    mlist = []  # We need to turn these regular arrays into matrix objects
    grd = np.zeros([3,3])
    for i in range(4):
        mlist.append(np.array([veclist[i]]))
        grd = grd + klist[i]*mlist[i].T #Fix this too!!
    return grd
#/////////////////////////////////////////////////////////////

#%%

def curv(veclist, klist):      # Modified version of the gradient calculation to get the curvature (1/Rc)
    i = 0
    mlist = []  # We need to turn these regular arrays into matrix objects
    vec = np.zeros(3)
    B_avg = sum(veclist)/4
    for i in range(4):
        mlist.append(np.array([veclist[i]]))
        
        # vec = vec + np.dot(mlist[i],klist[i]*mlist[i].T)/(np.linalg.norm(mlist[i])**2) # This version takes the 
            # Update: It seems that the sum of the 4 elements goes to near zero when using dot product
                #    as in, vec1 + vec3 + vec4 = - vec2, for any swapping of the numbers
                #    must be built in to the definition of the K vectors?? 
                #  I think this calculation essentially isolates everything to a sum of the Kvectors

        # vec = vec + mlist[i]*klist[i]*mlist[i].T/(np.linalg.norm(mlist[i])**2)   
        # This version gets close to believable units, but doesn't use a dot product 
        # Uses regular products, while still using matrices (so K*B is a Tensor)
        # Down to Rc ~ 50 km at max curvature 

        vec = vec + np.dot(B_avg/np.linalg.norm(B_avg),klist[i]*mlist[i].T/np.linalg.norm(mlist[i]))
        # This version is the best one that uses the correct dot product method and where K*B is a Tensor
        # "Fixed" K vector sum issue by using B_avg/|B_avg| for the first term in the dot product
        #      and k[i]B[i]/|B[i]| for the grad terms
        #  Down to about Rc ~ 100 km at max curvature   
        
    crv = np.linalg.norm(vec)    
    return crv

#%%
# Getting  Curl of B 
# Get all Bs and positions in np form and interpolted together
# [0,1,2,3] = MMS[1,2,3,4]
B_names, posinterp_names, pos_names,Binterp_names,Bflds,posits= [],[],[],[],[],[]

# Factors to convert fld & position to SI units (change fldfact depending on fld, keep posfact as is)
Bfact = 1e-9  
posfact = 1e3
# Get field and mec (position) data
for i in range(4):
    B_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  
    pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
    # tinterpol(B_names[i],B_names[i],newname = 'B' + str(i+1)) #interpolate
    tinterpol(pos_names[i],B_names[i],newname = 'pos' + str(i+1)) #interpolate
    # Binterp_names.append('B' + str(i+1))
    posinterp_names.append('pos' + str(i+1))
    Bflds.append(get_data(B_names[i]))
    posits.append(get_data(posinterp_names[i]))
# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = Bflds[0].times
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
    B_avg[i] =  (Bflds[0][i][:-1]+Bflds[1][i][:-1]+Bflds[2][i][:-1]+Bflds[3][i][:-1])/4   # Is this techincally correct?  Just a 4 pt avg
    klist = recip_vecs(posits[0][i],posits[1][i],posits[2][i],posits[3][i])
    grd[i] = grad(veclist,klist)
    crv[i] = curv(veclist,klist)
    Rc[i] = 1/crv[i]
    #Rc[i] = np.abs((np.dot(B_avg[i],grd[i]))**-1) / np.linalg.norm(B_avg)**2

#%%
store_data('curvature', data = {'x':timeax, 'y': 1e3*crv})
store_data('Rc', data = {'x':timeax, 'y': 1e-3*Rc})
# tsmooth('curvature', width=0.05*len(Rc),new_names = '1/R', preserve_nans=0)
options('curvature', 'thick',1.5)
options('Rc', 'thick',1.5)
options('Rc', 'ylog',1)
options('Rc', 'ytitle', 'Rc  [km]')
options('curvature', 'ytitle', 'Curvature  [1/km]')
# timespan('2017-08-10 12:18:00', 60, keyword='seconds')
# timespan('2017-07-11 22:33:30', 60, keyword='seconds')
#timespan('2015-10-16 13:06:50', 30,keyword='seconds')

#timespan('2017-06-17 20:23:50', 20,keyword='seconds')

#timespan('2017-06-17 20:23:50', 60, keyword='seconds')
#tplot(['Rc','curvature'])
# %%
# Next: Get gyroradii


fpi_vars = fpi(probe = probes,data_rate = 'brst',trange=trange,time_clip=True)
#%%

Te_names,Ti_names,B_names,Be_names,Bi_names= [],[],[],[],[]
for i in range(4):
    Te_names.append('mms' + str(probes[i]) + '_des_temptensor_gse_brst') #Change this if you want to calc for another vec quantity  
    Ti_names.append('mms' + str(probes[i]) + '_dis_temptensor_gse_brst') #Change this if you want to calc for another vec quantity  
    B_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  

edata = len(reform(get_data(Te_names[0])))
idata = len(reform(get_data(Ti_names[0])))
ed = np.zeros(ndata)
id = np.zeros(ndata)

Telist,Tilist,Blist,Relist,Rilist = [],[],[],[ed,ed,ed,ed],[id,id,id,id]
for sc in range(4):
    tinterpol(B_names[sc],Te_names[sc],newname='B')
    B = 1e-9*reform(get_data('B'))
    Te = e*reform(get_data(Te_names[sc]))
    Blist.append(B)
    Telist.append(Te)
    for i in range(len(B)):
        Relist[sc][i] = (me*np.sqrt(np.trace(Te[i])/me)/(e*np.linalg.norm(B[i])))
Re_avg = np.zeros(len(B))

for i in range(len(B)):
    Re_avg[i] = (Relist[0][i] + Relist[1][i] +Relist[2][i] +Relist[3][i])/4

for sc in range(4):
    tinterpol(B_names[sc],Ti_names[sc],newname='B')
    B = 1e-9*reform(get_data('B'))
    Ti = e*reform(get_data(Ti_names[sc]))
    Blist.append(B)
    Tilist.append(Ti)
    for i in range(len(B)):
        Rilist[sc][i] = (mi*np.sqrt(np.trace(Ti[i])/mi)/(e*np.linalg.norm(B[i])))
Ri_avg = np.zeros(len(B))

for i in range(len(B)):
    Ri_avg[i] = (Rilist[0][i] + Rilist[1][i] +Rilist[2][i] +Rilist[3][i])/4

store_data('Re_avg', data = {'x':get_data(Te_names[0]).times, 'y':1e-3*Re_avg})
store_data('Ri_avg', data = {'x':get_data(Ti_names[0]).times, 'y':1e-3*Ri_avg})

Rnames = ['Re_avg','Ri_avg']
options(Rnames, 'thick',1.5)
options(Rnames, 'ylog',1)
options('Re_avg', 'ytitle', 'Re  [km]')
options('Ri_avg', 'ytitle', 'Ri  [km]')
#%%
#tplot(['Rc','Re'])

# Rea = 1e3*reform(get_data('Re_avg'))
# Ria = 1e3*reform(get_data('Ri_avg'))
# Rcc = 1e3*reform(get_data('Rc'))

# tinterpol('Re_avg','Rc',newname='Rec')
# tinterpol('Ri_avg','Rc',newname='Ric')
tinterpol('Rc', 'Re_avg', newname='Rce')
tinterpol('Rc', 'Ri_avg', newname='Rci')

Rce = 1e3*reform(get_data('Rce')) #back to SI units
Rci = 1e3*reform(get_data('Rci'))
# Ric = 1e3*reform(get_data('Ric'))
# Rec = 1e3*reform(get_data('Rec'))
 
    

ecomp = np.zeros([len(Rce),2])
icomp = np.zeros([len(Rci),2])

for i in range(len(Rce)):
    ecomp[i] =  np.array([Rce[i],Re_avg[i]])

for i in range(len(Rci)):
    icomp[i] =  np.array([Rci[i],Ri_avg[i]])



store_data('Re/Rc', data = {'x': get_data(Te_names[0]).times, 'y': Re_avg/Rce})
store_data('Ri/Rc', data = {'x': get_data(Ti_names[0]).times, 'y': Ri_avg/Rci})
store_data('Rc & Re', data = {'x': get_data(Te_names[0]).times, 'y':1e-3*ecomp })
store_data('Rc & Ri', data = {'x': get_data(Ti_names[0]).times, 'y':1e-3*icomp })

names = ['Rc & Re','Rc & Ri']
ratio_names = ['Re/Rc','Ri/Rc']

options(ratio_names,'thick', 1.5)
options(names,'ylog', 1)
options(names,'thick', 1.5)
options('Rc & Re','ytitle', 'Rc and Re [km]')
options('Rc & Ri','ytitle', 'Rc and Ri [km]')
options('Rc & Re', 'Color', ['k','b'])
options('Rc & Ri', 'Color', ['k','r'])
options('Rc & Re','legend_names', ['Rc','Re'])
options('Rc & Ri','legend_names', ['Rc','Ri'])

# timespan('2017-08-10 12:18:00', 59, keyword='seconds')
# tplot('R comparison')

tplot(['Rc & Re','Rc & Ri','Re/Rc','Ri/Rc'])


# %%
