# Converts ps to png
from PIL import Image
import os



Choice = input("What is filename in filename.ps?       ")

def convert():
	img = Image.open(str(Choice) + '.ps')
	img.save(str(Choice) + '.png')


convert()


