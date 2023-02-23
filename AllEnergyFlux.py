# Script to obtain Energy fluxes for all spacecraft and get their Div/Curl

# Produces all quantities without error, but this is only a frst pass.  Might be some hidden issues

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
lmn_1209 = np.array([
[-0.091,0.87,0.49],
[-0.25,-0.49,0.83],
[0.96,-0.05,0.2700]])

I = np.identity(3)
#%%
# Get Data
trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
trange = ['2016-11-09/13:39:00', '2016-11-09/13:40:00']  # Shock.  Sep ~ 16 de
trange = ['2016-12-09/09:03:30', '2016-12-09/09:04:30']  # Electron. Sep ~ 5 de

#trange = ['2016-11-09/13:38:00', '2016-11-09/13:40:00']
#trange = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
# trange = ['2015-10-16/13:06:50', '2015-10-16/13:07:10']


probe  = [1,2,3,4] 
trange = trange

fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probe,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
mec_vars = mec(probe = probe,trange=trange,data_rate='srvy',time_clip=True)
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


# REMEMBER TO FIX!
def kin_flux(m,n,v):
	K = 0.5*m*n*v**3
	#K = 0.5*m*n*v*np.linalg.norm(v)**2# this uses |v|*v
	# K = v
	return K

# This works now even though its literally identical to the old version that didnt work..
def enth_flux(v,P):
	H = 0.5*v*np.trace(P) + np.dot(v,P)
	return H

def Poynt_flux(E,B):
	S = np.cross(E,B)/mu0
	return S

# Functions for Power densities (J.E, J.E', -PdotGrad(u), du/dt, Div(S))
def get_j(n,vi,ve):
	q = 1.6e-19 # charge unit in SI
	j = q*n*(vi - ve)
	return j

# Put a J.E function here!!
	
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

def get_Eprime(E,v,B):
	Ep = E + np.cross(v,B)
	return Ep




# Function to read data, interpolate, convert to SI units, and come out in the form of np arrays
# Cadence order (high -> low): edp & scm (same) -> fgm -> fpi-des -> fpi-dis
def AllFluxes(probe):
	frame = lmn_1209  #Keeping in GSE for now.  can convert to LMN at the end if desired
	# Field names variables
	B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
	E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
	vi_name = 'mms' + str(probe) + '_' + 'dis' + '_bulkv_gse_brst'
	ve_name = 'mms' + str(probe) + '_' + 'des' + '_bulkv_gse_brst'
	B_name = 'mms' + str(probe) + '_fgm_b_gse_brst_l2'
	E_name = 'mms' + str(probe) + '_edp_dce_gse_brst_l2'
	ne_name = 'mms' + str(probe) + '_' + 'des' + '_numberdensity_brst'
	ni_name = 'mms' + str(probe) + '_' + 'dis' + '_numberdensity_brst'
	Pi_name = 'mms' + str(probe) + '_dis_prestensor_gse_brst'
	Pe_name = 'mms' + str(probe) + '_des_prestensor_gse_brst'
	pos_name = 'mms'+ str(probe) + '_mec_r_gsm'
	var_name = B_name
	tinterpol(B_name,var_name, newname='B')
	tinterpol(E_name,var_name, newname='E')
	tinterpol(vi_name,var_name, newname='vi')
	tinterpol(ve_name,var_name, newname='ve')
	tinterpol(Pi_name,var_name, newname='Pi')
	tinterpol(Pe_name,var_name, newname='Pe')
	tinterpol(ni_name,var_name, newname='ni')
	tinterpol(ne_name,var_name, newname='ne')
	tinterpol(pos_name, var_name,newname='pos')
	B = 1e-9*reform(get_data('B'))
	E = 1e-3*reform(get_data('E'))
	vi = 1e3*reform(get_data('vi'))
	ve = 1e3*reform(get_data('ve'))
	ni = 1e6*reform(get_data('ni'))
	ne = 1e6*reform(get_data('ne'))
	Pi = 1e-9*reform(get_data('Pi'))
	Pe = 1e-9*reform(get_data('Pe'))
	ndata = len(vi)
	
	# Poynting Flux
	S = np.zeros_like(E)
	for i in range(ndata-1):
		S[i] = convert_lmn(Poynt_flux(E[i],B[i,:-1]),frame)

	# Electron Energy Flux
	Ke = np.zeros_like(E)
	He = np.zeros_like(E)
	for i in range(ndata-1):
		He[i] = convert_lmn(enth_flux(ve[i],Pe[i]),frame) #for some reason this works now
		#He[i] = convert_lmn(0.5*ve[i]*np.trace(Pe[i]) + np.dot(ve[i],Pe[i]),frame)
		Ke[i] = convert_lmn(kin_flux(me,ne[i],ve[i]),frame)  # this uses v^3  #unclear which is more correct. Eastwood uses v^3

	# Ion Energy Flux
	Ki = np.zeros_like(E)
	Hi = np.zeros_like(E)
	for i in range(ndata-1):
		Hi[i] = convert_lmn(enth_flux(vi[i],Pi[i]),frame)
		#Hi[i] = convert_lmn(0.5*vi[i]*np.trace(Pi[i]) + np.dot(vi[i],Pi[i]),frame)
		Ki[i] = convert_lmn(kin_flux(mi,ni[i],vi[i]),frame) 

	return S,He,Hi,Ke,Ki

S1,He1,Hi1,Ke1,Ki1 = AllFluxes(1)  
S2,He2,Hi2,Ke2,Ki2 = AllFluxes(2)  
S3,He3,Hi3,Ke3,Ki3 = AllFluxes(3)  
S4,He4,Hi4,Ke4,Ki4 = AllFluxes(4) 
ndata = len(S1) 


#%%

pos_names = [] # Names of positions 
fld_interp_names = [] # interpolated version of field names 
posits = [] # get_data for each position

# posfact = 1e3 


# Get field and mec (position) data
for i in range(4):
	pos_names.append('mms'+ str(probe[i]) + '_mec_r_gse')
	vi_name = 'mms' + str(probe[i]) + '_dis_bulkv_gse_brst'
	vi_name = 'mms3_fgm_b_gse_brst_l2_btot'
	tinterpol(pos_names[i],vi_name,newname = 'pos' + str(i+1)) #interpolate
	posits.append(get_data('pos' + str(i+1)))

# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = posits[0].times
timeax = get_data('mms1_dis_bulkv_gse_brst').times
timeax = get_data('mms3_fgm_b_gse_brst_l2_btot').times
ndata = len(timeax)
crl = np.zeros([ndata,3])
divr = np.zeros([ndata])


# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    posits[i] = 1e3*reform(posits[i]) # [pos1,pos2,pos3,pos4]

pos1,pos2,pos3,pos4 = posits[0],posits[1],posits[2],posits[3]
#%%
crl_S,crl_Hi,crl_He,crl_Ki,crl_Ke = np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3])
div_S,div_Hi,div_He,div_Ki,div_Ke = np.zeros([ndata]),np.zeros([ndata]),np.zeros([ndata]),np.zeros([ndata]),np.zeros([ndata])

# Get Div & Curl
for i in range(ndata-1):
	Slist = [S1[i],S2[i],S3[i],S4[i]]
	Hilist = [Hi1[i],Hi2[i],Hi3[i],Hi4[i]]
	Helist = [He1[i],He2[i],He3[i],He4[i]]
	Kilist = [Ki1[i],Ki2[i],Ki3[i],Ki4[i]]
	Kelist = [Ke1[i],Ke2[i],Ke3[i],Ke4[i]]
	klist = recip_vecs(pos1[i],pos2[i],pos3[i],pos4[i])


	crl_S[i] = curl(Slist,klist)
	div_S[i] = div(Slist,klist)

	crl_Hi[i] = curl(Hilist,klist)
	div_Hi[i] = div(Hilist,klist)
	
	crl_He[i] = curl(Helist,klist)
	div_He[i] = div(Helist,klist)

	crl_Ki[i] = curl(Kilist,klist)
	div_Ki[i] = div(Kilist,klist)

	crl_Ke[i] = curl(Kelist,klist)
	div_Ke[i] = div(Kelist,klist)





#%%
# Default units W/m^2
Slist = [S1,S2,S3,S4]
Hilist = [Hi1,Hi2,Hi3,Hi4]
Helist = [He1,He2,He3,He4]
Kilist = [Ki1,Ki2,Ki3,Ki4]
Kelist = [Ke1,Ke2,Ke3,Ke4]
names = []

for i in range(4):
	store_data('S'+str(i+1), data = {'x':timeax, 'y': Slist[i]})
	store_data('Ke'+str(i+1), data = {'x':timeax, 'y': Kelist[i]})
	store_data('He'+str(i+1), data = {'x':timeax, 'y': Helist[i]})
	store_data('Ki'+str(i+1), data = {'x':timeax, 'y': Kilist[i]})
	store_data('Hi'+str(i+1), data = {'x':timeax, 'y': Hilist[i]})
	names = names + ['S'+str(i+1), 'Ke'+str(i+1),'He'+str(i+1),'Ki'+str(i+1),'Hi'+str(i+1)]



#Why isn't MMS4 showing up in tplot?

store_data('divS', data = {'x':timeax, 'y': div_S})
store_data('divHi', data = {'x':timeax, 'y': div_Hi})
store_data('divHe', data = {'x':timeax, 'y': div_He})
store_data('divKi', data = {'x':timeax, 'y': div_Ki})
store_data('divKe', data = {'x':timeax, 'y': div_Ke})
divnames = ['divS','divHi','divHe','divKi','divKe']
options(divnames,'thick',1.5)
#options(divnames, 'yrange', [-1e-8,1e-8])
options(names, 'thick',1.5)
#tplot(names)
# # %%
# names = ['S','Ke','He','Ki','Hi']
# options(names, 'Color', ['b','g','r'])
# tplot_options('vertical_spacing',0.3)
# tplot(['S','Ke','He','Ki','Hi'])
# %%

from pyspedas.analysis.tsmooth import tsmooth
fld = 'divS'
tsmooth(fld, width=5,new_names = 'smoothed', preserve_nans=0)
options('smoothed', 'thick',1.5)

#timespan('2017-06-17 20:23:50', 60, keyword='seconds')
tplot([fld,'smoothed'])
# %%

# Need to get Curlometer J


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

probes = probe

# Get field and mec (position) data
for i in range(4):
    B_names.append('mms' + str(probes[i]) + '_fgm_b_gse_brst_l2') #Change this if you want to calc for another vec quantity  
    vi_names.append('mms' + str(probes[i]) + '_dis_bulkv_gse_brst') #Change this if you want to calc for another vec quantity  
    ve_names.append('mms' + str(probes[i]) + '_des_bulkv_gse_brst') #Change this if you want to calc for another vec quantity  
    ni_names.append('mms' + str(probes[i]) + '_dis_numberdensity_brst') #Change this if you want to calc for another vec quantity  
    ne_names.append('mms' + str(probes[i]) + '_des_numberdensity_brst') #Change this if you want to calc for another vec quantity  
    pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
    tinterpol(B_names[i],B_names[i],newname = 'B' + str(i+1)) #interpolate
    tinterpol(vi_names[i],B_names[i],newname = 'vi' + str(i+1)) #interpolate
    tinterpol(ve_names[i],B_names[i],newname = 've' + str(i+1)) #interpolate
    tinterpol(ni_names[i],B_names[i],newname = 'ni' + str(i+1)) #interpolate
    tinterpol(ne_names[i],B_names[i],newname = 'ne' + str(i+1)) #interpolate
    tinterpol(pos_names[i],B_names[i],newname = 'pos' + str(i+1)) #interpolate
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
    posits.append(get_data('pos' + str(i+1)))

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


Eflds = []
for i in range(4):
	tinterpol('mms' + str(probe[i]) + '_edp_dce_gse_brst_l2',B_names[i],newname = 'E' + str(i+1)) #interpolate
	Eflds.append(get_data('E' + str(i+1)))


for i in range(4):
    Eflds[i] = 1e3*reform(Eflds[i]) 

E_avg = sum(Eflds)

JdotE = np.zeros(len(E_avg))
for i in range(len(E_avg)):
	JdotE[i] = np.dot(j_crl[i],E_avg[i])

# timespan('2017-07-11 22:33:50', 30, keyword='seconds')

store_data('jdote', data = {'x':timeax,'y':JdotE})

options('jdote','thick', 1.5)

tplot(['divS','jdote'])

# Why are the units of jdote so ridiculous?? they're definitely way too large
# %%
