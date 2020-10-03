# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:36:44 2020

@author: Nils
"""

#%% imports
from skimage import io
from skimage import color
from skimage.filters import gaussian
from skimage.transform import resize


from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#import pandas as pd

import numpy as np


#from cv2 import GaussianBlur

#from gauss import difference_of_gaussians

#%% load image
path = r"../JenAesthetics/small/Giovanni_Francesco_Romanelli_-_" \
       "The_Finding_of_Moses_-_Google_Art_Project.jpg"
       

img_high = io.imread(path)

#%% low resolution

img = resize(img_high, (896, 1191))


#%% convert to lab
img_lab = color.rgb2lab(img)

#%% create colormaps
redgreen = LinearSegmentedColormap.from_list('a', ['green', 'white','red'])
yellowblue = LinearSegmentedColormap.from_list('b', ['blue','white', 'yellow'])



#%% plot full image

fig, ax = plt.subplots(figsize=(4, 5), dpi=300)

ax.imshow(img)
ax.axis('off')


#%% plot images
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), dpi=300)

#ax1.imshow(img)
#ax1.set_title("Original RGB")
#ax1.axis('off')

ax1.imshow(img_lab[:,:,0], cmap="Greys_r")
ax1.set_title("L channel")
ax1.axis('off')

ax2.imshow(img_lab[:,:,1], cmap=redgreen, vmin=-128, vmax=127)
ax2.set_title("a channel")
ax2.axis('off')

ax3.imshow(img_lab[:,:,2], cmap=yellowblue, vmin=-128, vmax=127)
ax3.set_title("b channel")
ax3.axis('off')


#%% Gaus Filter

''' Old 
sigmas = [2, 5, 10, 20, 50, 100]
sigmas.reverse()

gausfiltered = []
#gausfiltered = pd.DataFrame(index=sigmas, dtype='float64')

for sigma in sigmas:
    gausfiltered.append( gaussian(img_lab[:,:,0], sigma))
#imgg = gaussian(img_lab[:,:,0], 100)

'''


sigmas = [2, 5, 10, 20, 50, 100]
sigma_start = 10
sigma_range = 6
#sigmas = [2**x for x in range(sigma_start, sigma_range)]
gausfiltered = []

img_filtered = img_lab[:,:,0]


for i in range(sigma_range):
    img_filtered = gaussian(img_filtered, sigma_start)
    gausfiltered.append(img_filtered)


#%% plot filtered

fig, axs = plt.subplots(len(gausfiltered), 1, figsize=(1.3, 8), dpi=300)

for ax, gaus, sigma in zip(axs, gausfiltered, sigmas):
    ax.imshow(gaus, cmap='Greys_r')
    ax.set_title(str(sigma))
    ax.axis('off')


#%% differences

differences = np.diff(np.array(gausfiltered), n=1, axis=0)


#%%

fig, axs = plt.subplots(len(differences), 1, figsize=(1.3, 8), dpi=300)

for ax, diff in zip(axs, differences):
    ax.imshow(diff, cmap='Greys_r')
    ax.axis('off')

#%%
#diff = difference_of_gaussians(img_lab[:,:,0], 100)
#io.imshow(diff, cmap='Greys_r')



#%% cv2 Gaus

#imggcv = GaussianBlur(img_lab[:,:,0], (401,401), 100)
#io.imshow(imggcv, cmap='Greys_r')