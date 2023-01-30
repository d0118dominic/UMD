# Converts ps to png
#%%
from PIL import Image
import os
import glob

files = glob.glob('figs/*ps')

for i in range(len(files)):
	img = Image.open(files[i])
	img.save(str(files[i][:-3]) + '.png')
	print(files[i])
