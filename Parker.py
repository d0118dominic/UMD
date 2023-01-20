# Notebook for Parker data

#%%
import pyspedas
from pytplot import tplot


trange = ['2018-11-05', '2018-11-05/06:00']
fields_vars = pyspedas.psp.fields(trange=['2021-11-20', '2021 -11-21'], 
    datatype='mag_rtn', level='l2', time_clip=True)
tplot('psp_fld_l2_mag_RTN')


