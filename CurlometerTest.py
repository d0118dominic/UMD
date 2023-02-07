# Compare 2 different methods of calculating barycentric current density

# 1) Curlometer Technique: use J = Curl(B)/mu0
# 2) Moments Technique: Use J = average J = qn(vi-ve)

# A similar result between methods (1) and (2) suggest that spatial derivative approximations with MMS are reasonably trustworthy



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
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
#trange = ['2015-12-08/11:27:00','2015-12-08/11:30:00']
#trange = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
mec_vars = mec(probe = probes,trange=trange,data_rate='brst',time_clip=True)
fgm_vars = fgm(probe = probes, data_rate = 'brst', trange=trange,time_clip=True)
fpi_vars = fpi(probe = probes,data_rate = 'brst',trange=trange,time_clip=True)
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
# Interp function currently not in use

def get_j(n,vi,ve):
	q = 1.6e-19 # charge unit in SI
	j = q*n*(vi - ve)
	return j
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



def curl(veclist, klist):
    i = 0
    crl = np.array([0,0,0])
    for i in range(4):
        crl = crl + np.cross(klist[i],veclist[i])
    return crl

# Getting  Curl of B 
# Get all Bs and positions in np form and interpolted together
# [0,1,2,3] = MMS[1,2,3,4]
B_names, vi_names,ve_names,ni_names,ne_names,pos_names,\
Binterp_names, viinterp_names,veinterp_names,niinterp_names,neinterp_names,\
Bflds, viflds,veflds, neflds, niflds, posits\
= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

# Factors to convert fld & position to SI units (change fldfact depending on fld, keep posfact as is)
Bfact = 1e-9  
vfact = 1e3 
nfact = 1e6  
posfact = 1e3 


# Get field and mec (position) data
for i in range(4):
    B_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  
    vi_names.append('mms' + str(probes[i]) + '_dis_bulkv_gse_brst') #Change this if you want to calc for another vec quantity  
    ve_names.append('mms' + str(probes[i]) + '_des_bulkv_gse_brst') #Change this if you want to calc for another vec quantity  
    ni_names.append('mms' + str(probes[i]) + '_dis_numberdensity_brst') #Change this if you want to calc for another vec quantity  
    ne_names.append('mms' + str(probes[i]) + '_des_numberdensity_brst') #Change this if you want to calc for another vec quantity  
    pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
    tinterpol(B_names[i],pos_names[i],newname = 'B' + str(i+1)) #interpolate
    tinterpol(vi_names[i],pos_names[i],newname = 'vi' + str(i+1)) #interpolate
    tinterpol(ve_names[i],pos_names[i],newname = 've' + str(i+1)) #interpolate
    tinterpol(ni_names[i],pos_names[i],newname = 'ni' + str(i+1)) #interpolate
    tinterpol(ne_names[i],pos_names[i],newname = 'ne' + str(i+1)) #interpolate
    Binterp_names.append('B' + str(i+1))
    viinterp_names.append('vi' + str(i+1))
    veinterp_names.append('ve' + str(i+1))
    niinterp_names.append('ni' + str(i+1))
    neinterp_names.append('ne' + str(i+1))
    Bflds.append(get_data(Binterp_names[i]))
    viflds.append(get_data(viinterp_names[i]))
    veflds.append(get_data(veinterp_names[i]))
    niflds.append(get_data(niinterp_names[i]))
    neflds.append(get_data(neinterp_names[i]))
    posits.append(get_data(pos_names[i]))

# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = Bflds[0].times
ndata = len(timeax)
crl = np.zeros([ndata,3])


# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    Bflds[i] = Bfact*reform(Bflds[i]) # [fld1,fld2,fld3,fld4]
    viflds[i] = vfact*reform(viflds[i]) # [fld1,fld2,fld3,fld4]
    veflds[i] = vfact*reform(veflds[i]) # [fld1,fld2,fld3,fld4]
    niflds[i] = nfact*reform(niflds[i]) # [fld1,fld2,fld3,fld4]
    neflds[i] = nfact*reform(neflds[i]) # [fld1,fld2,fld3,fld4]
    posits[i] = posfact*reform(posits[i]) # [pos1,pos2,pos3,pos4]


# If fld is 4D (usually because total is included), chop off the 4th term
if len(Bflds[0][0]) == 4:   # Just put this here so its convenient to minimize in vscode
    Bfld1 = np.zeros([ndata,3])
    Bfld2 = np.zeros([ndata,3])
    Bfld3 = np.zeros([ndata,3])
    Bfld4 = np.zeros([ndata,3])
    for i in range(ndata-1):
        Bfld1[i] = Bflds[0][i][:-1]
        Bfld2[i] = Bflds[1][i][:-1] 
        Bfld3[i] = Bflds[2][i][:-1]
        Bfld4[i] = Bflds[3][i][:-1]
    Bflds = [Bfld1,Bfld2,Bfld3,Bfld4]



## Get Curlometer J ##
for i in range(ndata-1):
    veclist = [Bflds[0][i],Bflds[1][i],Bflds[2][i],Bflds[3][i]]
    klist = recip_vecs(posits[0][i],posits[1][i],posits[2][i],posits[3][i])
    crl[i] = curl(veclist,klist)

j_crl = crl/mu0

## Calculate 'moments' J ##
j = np.zeros_like(veflds)
j_mom = np.zeros_like(veflds[0])

for i in range(ndata-1):
    for sc in range(4):
        j[sc,i] = get_j(niflds[sc][i],viflds[sc][i],veflds[sc][i])
    
    j_mom[i] = functools.reduce(lambda x,y:x+y,j[:,i])/4

#%%
err = np.zeros_like(j_mom)
diff = np.zeros_like(err)

def percent_err(a,b):
    err = np.abs(200*np.abs(a-b)/(a+b))
    return err

for i in range(ndata-1):
    err[i] = percent_err(j_crl[i],j_mom[i])
    diff[i] = np.abs(j_crl[i]-j_mom[i])


for i in range(ndata-1):
    if np.any(np.isnan(err[i])):
        err[i] = 1e-15

#%%
# If you want to convert to more convenient non-SI units, do so here.
tnames = ['J_crl','J_mom', '|difference|']

tplot_options('y_range',[-2e-7,2e-7])
store_data('J_crl', data = {'x':timeax, 'y': j_crl})
store_data('J_mom', data = {'x':timeax, 'y': j_mom})
store_data('percent error', data = {'x':timeax, 'y': err})
store_data('|difference|', data = {'x':timeax, 'y': diff})
options(tnames, 'Color', ['b','g','r'])
options(tnames, 'yrange', [-0.5e-7,2e-7])
options('percent error', 'yrange', [0,1000])
options('J_crl', 'ytitle', 'Curl(B)/mu0')
options('J_mom', 'ytitle', '< J >')
options(tnames,'legend_names', ['x','y','z'])
tplot(tnames)
 # %%
