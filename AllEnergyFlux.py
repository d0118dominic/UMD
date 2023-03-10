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
from pytplot import get_data, store_data,timespan
from pyspedas.analysis.tsmooth import tsmooth
from matplotlib.pyplot import plot
from matplotlib.pyplot import scatter
me = 9.1094e-31 #kg
mi = 1837*me
mu0 = 1.2566370e-06  #;m kg / C^2
eps0 = 8.85e-12   # C^2/Nm^2

lmn_1209 = np.array([
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


frame = lmn_1209  # Choose the LMN coordinate system you want to use.  Choose I to keep GSE

# Get Data
# trange = ['2017-08-10/12:18:00', '2017-08-10/12:19:00']
# timespan('2017-08-10 12:18:20', 20, keyword='seconds')


# trange = ['2017-07-11/22:33:30', '2017-07-11/22:34:30']
# timespan('2017-07-11 22:34:00', 5, keyword='seconds')


# Hubbert Events (aside from June 17th)
# trange = ['2017-07-20/09:59:00 2017-07-20/09:10:00']
# trange = ['2017-06-19/09:43:00', '2017-06-19/09:45:00']


# trange = ['2016-11-09/13:39:00', '2016-11-09/13:40:00']  # Shock.  Sep ~ 16 de


trange = ['2016-12-09/09:03:30', '2016-12-09/09:04:30']  # Electron. Sep ~ 5 de
timespan('2016-12-09 09:03:54', 0.6, keyword='seconds')


#trange = ['2016-11-09/13:38:00', '2016-11-09/13:40:00']
# trange = ['2017-06-17/20:23:30', '2017-06-17/20:24:30']
#timespan('2017-06-17 20:24:05', 5, keyword='seconds')

# trange = ['2017-06-17/20:24:05', '2017-06-17/20:24:10']

# trange = ['2015-10-16/13:06:30', '2015-10-16/13:07:20']
# timespan('2015-10-16 13:06:50', 20, keyword='seconds')

probe  = [1,2,3,4] 
trange = trange

#%%
fgm_vars = fgm(probe = probe, data_rate = 'brst', trange=trange,time_clip=True)
edp_vars = edp(probe = probe,data_rate = 'brst',trange=trange,time_clip=True) 
fpi_vars = fpi(probe = probe,data_rate = 'brst',trange=trange,time_clip=True)
mec_vars = mec(probe = probe,trange=trange,data_rate='srvy',time_clip=True)
#%%

interpfld = 'mms1_fgm_b_gse_brst_l2'
# interpfld = 'mms1_dis_numberdensity_brst'
# interpfld = 'mms1_des_numberdensity_brst'
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

def Eprime(E,v,B):
	Ep = E + np.cross(v,B)
	return Ep
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
	return ue


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

# Energy Flux Transport
# Thermal
def Therm_Transport(v,P):
	H = enth_flux(v,P)
	u = therm_dens(P)
	tft = H/u
	return tft

def Kin_Transport(m,n,v):
	K = kin_flux(m,n,v)
	u = kinetic_dens(m,n,v)
	kft = K/u
	return kft

def EM_Transport(E,B):
	S = Poynt_flux(E,B)
	uB = B_dens(B)
	uE = E_dens(E)
	emft = S/(uB+uE)
	return emft


# Add MFT 
def MFT(E,B):
	pass
	return None
#


# 
#  



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

    # Get each k vector and put in a list  (Also convert Kvec to LMN coord)
    k1 = convert_lmn(kvec(pos23,pos24,pos21),frame)
    k2 = convert_lmn(kvec(pos34,pos31,pos32),frame)
    k3 = convert_lmn(kvec(pos41,pos42,pos43),frame)
    k4 = convert_lmn(kvec(pos12,pos13,pos14),frame)

    klist = np.array([k1,k2,k3,k4])

    return klist


# Define divergence given a veclist and klist
# where veclist is a list of some vector quantity measured at [MMS1,MMS2,MMS3,MMS4]
# and klist is the list of reciprocal vectors [k1,k2,k3,k4]
def grad(veclist, klist):
    i = 0
    mlist = []  # We need to turn these regular arrays into matrix objects
    grd = np.zeros([3,3])
    for i in range(4):
        mlist.append(np.array([veclist[i]]))
        grd = grd + klist[i]*mlist[i].T #Fix this too!!
    return grd

def grad_scalar(scalist, klist):
    i = 0
    grd = np.zeros(3)
    for i in range(4):
        grd = grd + klist[i]*scalist[i] #Fix this too!!
    return convert_lmn(grd,frame)

def div(veclist, klist):
	i = 0
	div,divx,divy,divz = 0,0,0,0
	for i in range(4):
		divx = divx + klist[i,0]*veclist[i][0]
		divy = divy + klist[i,1]*veclist[i][1]
		divz = divz + klist[i,2]*veclist[i][2]
		div = div + np.dot(klist[i],veclist[i])
	div = divx+divy+divz
	return np.array([divx,divy,divz,div])

def curl(veclist, klist):
    i = 0
    crl = np.array([0,0,0])
    for i in range(4):
        crl = crl + np.cross(klist[i],veclist[i])
    return convert_lmn(crl,frame)

def PStrain(P,vlist,klist):
	i = 0
	Pst = np.dot(P,grad(vlist,klist)) # This should use the average 
	return Pst





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





##%%


def AllDensities(probe):
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
	pos_name = 'mms'+ str(probe) + '_mec_r_gse'
	var_name = interpfld
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
	
	# EM Energy Density
	uE = np.zeros_like(ni)
	uB = np.zeros_like(ni)
	uem = np.zeros_like(ni)
	for i in range(ndata-1):
		uE[i] = E_dens(E[i])  #Array element sequence error?
		uB[i] = B_dens(B[i])
		uem[i] = uE[i] + uB[i]
	
	# Kinetic Energy Density
	uke = np.zeros_like(ni)
	uki = np.zeros_like(ni)
	uk = np.zeros_like(ni)
	for i in range(ndata-1):
		uke[i] = kinetic_dens(me,ne[i],ve[i])
		uki[i] = kinetic_dens(mi,ni[i],vi[i])
		uk[i] = uke[i] + uki[i]

	# Thermal Energy Density
	ute = np.zeros_like(ni)
	uti = np.zeros_like(ni)
	ut = np.zeros_like(ni)
	for i in range(ndata-1):
		ute[i] = therm_dens(Pe[i])
		uti[i] = therm_dens(Pi[i])
		ut[i] = ute[i] + uti[i]

	# Plasma Energy Density
	ue = np.zeros_like(ni)
	ui = np.zeros_like(ni)
	up = np.zeros_like(ni)
	for i in range(ndata-1):
		ue[i] = ute[i] + uke[i]
		ui[i] = uti[i] + uki[i]
		up[i] = ue[i] + ui[i]
	return uem,uE,uB,up,ue,ui,ut,ute,uti,uk,uke,uki

# Function to read data, interpolate, convert to SI units, and come out in the form of np arrays
# Cadence order (high -> low): edp & scm (same) -> fgm -> fpi-des -> fpi-dis
def AllFluxes(probe):
	# frame = lmn_1209  # convert to LMN at the end if desired
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
	var_name = interpfld
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

    ###########
	# Use ion frame E
	# Ep = np.zeros_like(E)
	# for i in range(ndata-1):
	# 	Ep[i] = Eprime(E[i],vi[i],B[i][:-1])
	# E = Ep
    ############


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

#//////////////////////////////////////////////////////////////////////////////
# All Divs for energy flux divergences
def AllDivs():
	pos_names = [] # Names of positions 
	posits = [] # get_data for each position
	# Get field and mec (position) data
	for i in range(4):
		pos_names.append('mms'+ str(probe[i]) + '_mec_r_gse')
		tinterpol(pos_names[i],interpfld,newname = 'pos' + str(i+1)) #interpolate
		posits.append(get_data('pos' + str(i+1)))
	timeax = posits[0].times
	ndata = len(timeax)
	
	# Reform data into np arrays in SI units (flds and posits)
	for i in range(4):
		posits[i] = 1e3*reform(posits[i]) # [pos1,pos2,pos3,pos4]
	pos1,pos2,pos3,pos4 = posits[0],posits[1],posits[2],posits[3]

	# Get Divs
	div_S,div_Hi,div_He,div_Ki,div_Ke = np.zeros([ndata,4]),np.zeros([ndata,4]),np.zeros([ndata,4]),np.zeros([ndata,4]),np.zeros([ndata,4])
	for i in range(ndata-1):
		Slist = [S1[i],S2[i],S3[i],S4[i]]
		Hilist = [Hi1[i],Hi2[i],Hi3[i],Hi4[i]]
		Helist = [He1[i],He2[i],He3[i],He4[i]]
		Kilist = [Ki1[i],Ki2[i],Ki3[i],Ki4[i]]
		Kelist = [Ke1[i],Ke2[i],Ke3[i],Ke4[i]]
		klist = recip_vecs(pos1[i],pos2[i],pos3[i],pos4[i])

		div_S[i] = div(Slist,klist)
		div_Hi[i] = div(Hilist,klist)
		div_He[i] = div(Helist,klist)
		div_Ki[i] = div(Kilist,klist)
		div_Ke[i] = div(Kelist,klist)

	return div_S,div_He,div_Hi,div_Ke,div_Ki


def AllGrads():
	pos_names = [] # Names of positions 
	posits = [] # get_data for each position
	# Get field and mec (position) data
	for i in range(4):
		pos_names.append('mms'+ str(probe[i]) + '_mec_r_gse')
		tinterpol(pos_names[i],interpfld,newname = 'pos' + str(i+1)) #interpolate
		posits.append(get_data('pos' + str(i+1)))
	timeax = posits[0].times
	ndata = len(timeax)
	
	# Reform data into np arrays in SI units (flds and posits)
	for i in range(4):
		posits[i] = 1e3*reform(posits[i]) # [pos1,pos2,pos3,pos4]
	pos1,pos2,pos3,pos4 = posits[0],posits[1],posits[2],posits[3]

	
	# Get Grads
	grd_uem,grd_up,grd_ue,grd_ui,grd_ute = np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3]),np.zeros([ndata,3])
	for i in range(ndata-1):
		uemlist = [uem1[i],uem2[i],uem3[i],uem4[i]]
		uplist = [up1[i],up2[i],up3[i],up4[i]]
		uelist = [ue1[i],ue2[i],ue3[i],ue4[i]]
		uilist = [ui1[i],ui2[i],ui3[i],ui4[i]]
		utelist = [ute1[i],ute2[i],ute3[i],ute4[i]]
		klist = recip_vecs(pos1[i],pos2[i],pos3[i],pos4[i])

		grd_uem[i] =  grad_scalar(uemlist,klist)
		grd_up[i] = grad_scalar(uplist,klist)
		grd_ue[i] = grad_scalar(uelist,klist)
		grd_ui[i] = grad_scalar(uilist,klist)
		grd_ute[i] = grad_scalar(utelist,klist)

	return grd_uem,grd_up,grd_ue,grd_ui,grd_ute

#%%
S1,He1,Hi1,Ke1,Ki1 = AllFluxes(1)  
S2,He2,Hi2,Ke2,Ki2 = AllFluxes(2)  
S3,He3,Hi3,Ke3,Ki3 = AllFluxes(3)  
S4,He4,Hi4,Ke4,Ki4 = AllFluxes(4) 
uem1,uE1,uB1,up1,ue1,ui1,ut1,ute1,uti1,uk1,uke1,uki1 = AllDensities(1)
uem2,uE2,uB2,up2,ue2,ui2,ut2,ute2,uti2,uk2,uke2,uki2 = AllDensities(2)
uem3,uE3,uB3,up3,ue3,ui3,ut3,ute3,uti3,uk3,uke3,uki3 = AllDensities(3)
uem4,uE4,uB4,up4,ue4,ui4,ut4,ute4,uti4,uk4,uke4,uki4 = AllDensities(4)
div_S,div_He,div_Hi,div_Ke,div_Ki = AllDivs()
grd_uem,grd_up,grd_ue,grd_ui,grd_ute = AllGrads()

ndata = len(S1)

pos_names = [] # Names of positions 
fld_interp_names = [] # interpolated version of field names 
posits = [] # get_data for each position

# Get field and mec (position) data
for i in range(4):
	pos_names.append('mms'+ str(probe[i]) + '_mec_r_gse')
	tinterpol(pos_names[i],interpfld,newname = 'pos' + str(i+1)) #interpolate
	posits.append(get_data('pos' + str(i+1)))

# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
timeax = posits[0].times
ndata = len(timeax)
crl = np.zeros([ndata,3])
divr = np.zeros([ndata])

Slist = [S1,S2,S3,S4]
Hilist = [Hi1,Hi2,Hi3,Hi4]
Helist = [He1,He2,He3,He4]
Kilist = [Ki1,Ki2,Ki3,Ki4]
Kelist = [Ke1,Ke2,Ke3,Ke4]
names = []
for i in range(4):
	store_data('S'+str(i+1), data = {'x':timeax, 'y': 1e9*Slist[i]})
	store_data('Ke'+str(i+1), data = {'x':timeax, 'y': 1e9*Kelist[i]})
	store_data('He'+str(i+1), data = {'x':timeax, 'y': 1e9*Helist[i]})
	store_data('Ki'+str(i+1), data = {'x':timeax, 'y': 1e9*Kilist[i]})
	store_data('Hi'+str(i+1), data = {'x':timeax, 'y': 1e9*Hilist[i]})
	names = names + ['S'+str(i+1), 'Ke'+str(i+1),'He'+str(i+1),'Ki'+str(i+1),'Hi'+str(i+1)]

uemlist = [uem1,uem2,uem3,uem4]
uplist = [up1,up2,up3,up4]
uelist = [ue1,ue2,ue3,ue4]
uilist = [ui1,ui2,ui3,ui4]
utelist = [ute1,ute2,ute3,ute4]
store_data('uem', data = {'x':timeax, 'y': 0.25e9*(uemlist[0]+uemlist[1]+uemlist[2]+uemlist[3])})
store_data('up', data = {'x':timeax, 'y': 0.25e9*(uplist[0]+uplist[1]+uplist[2]+uplist[3])})
store_data('ue', data = {'x':timeax, 'y': 0.25e9*(uelist[0]+uelist[1]+uelist[2]+uelist[3])})
store_data('ui', data = {'x':timeax, 'y': 0.25e9*(uilist[0]+uilist[1]+uilist[2]+uilist[3])})
store_data('ute', data = {'x':timeax, 'y': 0.25e9*(utelist[0]+utelist[1]+utelist[2]+utelist[3])})
unames = ['uem','up','ue','ui','ute']

store_data('divS', data = {'x':timeax, 'y': 1e9*div_S})     # ALL CONVERTED TO nW/m^3
store_data('divHi', data = {'x':timeax, 'y': 1e9*div_Hi})
store_data('divHe', data = {'x':timeax, 'y': 1e9*div_He})
store_data('divKi', data = {'x':timeax, 'y': 1e9*div_Ki})
store_data('divKe', data = {'x':timeax, 'y': 1e9*div_Ke})
divnames = ['divS','divHi','divHe','divKi','divKe']

store_data('grd_uem', data = {'x':timeax,'y': 1e9*grd_uem})
store_data('grd_up', data = {'x':timeax,'y': 1e9*grd_up})
store_data('grd_ue', data = {'x':timeax,'y': 1e9*grd_ue})
store_data('grd_ui', data = {'x':timeax,'y': 1e9*grd_ui})
store_data('grd_ute', data = {'x':timeax,'y': 1e9*grd_ute})
grdnames = ['grd_uem','grd_up','grd_ue','grd_ui','grd_ute']

#%%
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
	pos_names.append('mms'+ str(probes[i]) + '_mec_r_gsm')
	# tinterpol(B_names[i],B_names[i],newname = 'B' + str(i+1)) #interpolate
	# tinterpol(pos_names[i],B_names[i],newname = 'pos' + str(i+1)) #interpolate
	tinterpol(B_names[i],interpfld,newname = 'B' + str(i+1))
	tinterpol(pos_names[i],interpfld,newname = 'pos' + str(i+1))
	Bflds.append(get_data('B' + str(i+1)))
	posits.append(get_data('pos' + str(i+1)))

# N data points and time axis (setting to different vars here bc flds will change form)
# Also define shape of curl vs divergence (vector vs scalar)
# a = get_data(interpfld)
# timeax = a.times
# ndata = len(timeax)
# crl = np.zeros([ndata,3])
timeax = Bflds[0].times
ndata = len(timeax)
crl = np.zeros([ndata,3])

# Reform data into np arrays in SI units (flds and posits)
for i in range(4):
    Bflds[i] = Bfact*reform(Bflds[i]) # [fld1,fld2,fld3,fld4]
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
	tinterpol('mms' + str(probe[i]) + '_edp_dce_gse_brst_l2',interpfld,newname = 'E' + str(i+1)) #interpolate
	Eflds.append(get_data('E' + str(i+1)))


for i in range(4):
    Eflds[i] = 1e-3*reform(Eflds[i]) 



# temporary fix for 8/10 event:
# np.concatenate(Eflds[1],Eflds[1][-2])
##########################
E_avg = sum(Eflds)/4

JdotE = np.zeros(len(E_avg))
for i in range(len(E_avg)):
	JdotE[i] = np.dot(j_crl[i],E_avg[i])


# timespan('2016-12-09 09:03:54', 1, keyword='seconds')

# store_data('Poynt', data = {'x':timeax,'y':1e9*np.array([JdotE,div_S,-JdotE-div_S])})
store_data('jdote', data = {'x':timeax,'y':1e9*JdotE})  # CONVERTED TO nW/m^3

store_data('du/dt', data = {'x':timeax,'y': -1e9*(JdotE+div_S[:,3])})


#%%
pterms = ['divS','jdote','du/dt','Poynt Terms']
terms = ['divS','jdote','du/dt','divHe','divKe']
options(pterms,'thick', 1.5)

divterms = ['divS','divHe','divKe','divHi','divKi']
options(divterms,'Color',['b','g','r','k'])
#options(divterms,'legend_names', ['L','M','N','Total'])
options(divterms,'thick',1)

# options(terms,'yrange',[-8,8])
options(terms,'yrange',[-200,200])
# options(pterms,'yrange',[-1,1])
width = 0
tsmooth('divS', width=width,new_names = 'div_S', preserve_nans=0)
tsmooth('jdote', width=width,new_names = 'j_dot_e', preserve_nans=0)
tsmooth('du/dt', width=width,new_names = 'du_dt', preserve_nans=0)
tplot(['j_dot_e','div_S','du_dt','divHe','divKe'])
#tplot(['jdote','divS','du/dt','divHe'])

# Clean up interp schemes bc j.e looks a little larger now for the 12/09 evbent
# %%
