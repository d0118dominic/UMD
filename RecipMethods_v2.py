
# Use pyspedas to do the multi-spacecrft calculations like divergence & curl
#   (work in progress)

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

# Define trange and get mec data from all 4 spacecraft
probes = [1,2,3,4]
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
#trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
mec_vars = mec(probe = probes,trange=trange,data_rate='brst',time_clip=True)
fgm_vars = fgm(probe = probes, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probes,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probes,data_rate = 'brst',trange=trange,time_clip=True)


#%%
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

def interp_to(var_name):
    tinterpol(B_name,var_name, newname='B')
    tinterpol(E_name,var_name, newname='E')
    tinterpol(vi_name,var_name, newname='vi')
    tinterpol(ve_name,var_name, newname='ve')
    tinterpol(Pi_name,var_name, newname='Pi')
    tinterpol(Pe_name,var_name, newname='Pe')
    tinterpol(ni_name,var_name, newname='ni')
    tinterpol(ne_name,var_name, newname='ne')
    tinterpol(pos_name,var_name,newname = 'pos')

    B = 1e-9*reform(get_data('B'))
    E = 1e-3*reform(get_data('E'))
    vi = 1e3*reform(get_data('vi'))
    ve = 1e3*reform(get_data('ve'))
    ni = 1e6*reform(get_data('ni'))
    ne = 1e6*reform(get_data('ne'))
    Pi = 1e-9*reform(get_data('Pi'))
    Pe = 1e-9*reform(get_data('Pe'))
    pos = 1e3*reform(get_data('pos'))

    ndata = len(B)

    return B,E,vi,ve,ni,ne,Pi,Pe,pos,ndata

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

# Field names variables
def get_names(probe):
    B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
    E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
    vi_name = 'mms' + str(probe) + '_dis_bulkv_gse_brst'
    ve_name = 'mms' + str(probe) + '_des_bulkv_gse_brst'
    ne_name = 'mms' + str(probe) + '_des_numberdensity_brst'
    ni_name = 'mms' + str(probe) + '_dis_numberdensity_brst'
    Pi_name = 'mms' + str(probe) + '_dis_prestensor_gse_brst'
    Pe_name = 'mms' + str(probe) + '_des_prestensor_gse_brst'
    pos_name = 'mms'+ str(probe) + '_mec_r_gsm'
    return B_name, E_name, vi_name, ve_name, ni_name, ne_name, Pi_name, Pe_name, pos_name


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

def get_lists():
    Blist,Elist,vilist,velist,Pilist,Pelist,poslist = [],[],[],[],[],[],[]
    i=1
    while i<5:

        B_name,E_name,vi_name,ve_name,ni_name,ne_name,Pi_name,Pe_name,pos_name=get_names(i)
        B,E,vi,ve,ni,ne,Pi,Pe,pos,ndata = interp_to(pos_name)
        Blist.append(B)
        Elist.append(E)
        vilist.append(vi)
        velist.append(ve)
        Pilist.append(Pi)
        Pelist.append(Pe)
        poslist.append(pos)
        i+=1
    

    return Blist,Elist,vilist,velist,Pilist,Pelist,poslist,ndata

 
#%%

#Still not working right
#Issue comes from calculation of kvectors
Blists,Elists,vilists,velists,Pilists,Pelists,poslists,ndata = get_lists()
crlB = np.zeros([ndata,3])
divB = np.zeros([ndata])
for i in range(ndata-1):
    Blist = [Blists[0][i][:-1],Blists[1][i][:-1],Blists[2][i][:-1],Blists[3][i][:-1]]
    Elist = [Elists[0][i],Elists[1][i],Elists[2][i],Elists[3][i]]
    vilist = [vilists[0][i],vilists[1][i],vilists[2][i],vilists[3][i]]
    velist = [velists[0][i],velists[1][i],velists[2][i],velists[3][i]]
    Pilist = [Pilists[0][i],Pilists[1][i],Pilists[2][i],Pilists[3][i]]
    Pelist = [Pelists[0][i],Pelists[1][i],Pelists[2][i],Pelists[3][i]]
    klist = recip_vecs(poslists[0][i],poslists[1][i],poslists[2][i],poslists[3][i]) #this is producing nan values
    crlB[i] = curl(Blist,klist)
    divB[i] = div(Blist,klist)



#%%



#%%



#%%

#%%






#%%


# B1_name,E1_name,vi1_name,ve1_name,ni1_name,ne1_name,Pi1_name,Pe1_name,pos1_name=get_names(probes[0])
# B2_name,E2_name,vi2_name,ve2_name,ni2_name,ne2_name,Pi2_name,Pe2_name,pos2_name=get_names(probes[1])
# B3_name,E3_name,vi3_name,ve3_name,ni3_name,ne1_name,Pi3_name,Pe3_name,pos3_name=get_names(probes[2])
# B4_name,E4_name,vi4_name,ve4_name,ni4_name,ne1_name,Pi4_name,Pe4_name,pos4_name=get_names(probes[3])
# Bnames = [B1_name,B2_name,B3_name,B4_name] 
# Enames = [E1_name,E2_name,E3_name,E4_name] 
# vinames = [vi1_name,vi2_name,vi3_name,vi4_name] 
# venames = [ve1_name,ve2_name,ve3_name,ve4_name] 
# Pinames = [Pi1_name,Pi2_name,Pi3_name,Pi4_name] 
# Penames = [Pe1_name,Pe2_name,Pe3_name,Pe4_name] 
# posnames = [pos1_name,pos2_name,pos3_name,pos4_name]
# for i in range(4):
#     tinterpol(Bnames[i],posnames[i], 'B' + str(i+1))
#     tinterpol(Enames[i],posnames[i], 'E' + str(i+1))
#     tinterpol(vinames[i],posnames[i], 'vi' + str(i+1))
#     tinterpol(venames[i],posnames[i], 've' + str(i+1))
#     tinterpol(Pinames[i],posnames[i], 'Pi' + str(i+1))
#     tinterpol(Penames[i],posnames[i], 'Pe' + str(i+1))
#    ## TBD 
# Bnames = ['B1','B2','B3','B4']
# Enames = ['E1','E2','E3','E4']
# vinames = ['vi1','vi2','vi3','vi4']
# venames = ['ve1','ve2','ve3','ve4']
# Pinames = ['Pi1','Pi2','Pi3','Pi4']
# Penames = ['Pe1','Pe2','Pe3','Pe4']
# Blist,Elist,vilist,velist,Pilist,Pelist,poslist = [],[],[],[],[],[],[]
# for i in range(4):
#     Blist.append(get_data(Bnames[i]))
#     Elist.append(get_data(Enames[i]))
#     vilist.append(get_data(vinames[i]))
#     velist.append(get_data(venames[i]))
#     Pilist.append(get_data(Pinames[i]))
#     Pelist.append(get_data(Penames[i]))
#     poslist.append(get_data(posnames[i]))
#ion = get_data(ni_name)
#elec = get_data(ne_name)
#Bfld = get_data(B_name)
#Efld = get_data(E_name)