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
from skimage.feature import hog

from skimage import exposure

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import math

import timeit

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


#%% Gaus Filter old

'''
sigmas = [2, 5, 10, 20, 50, 100]
#sigmas.reverse()

gausfiltered = []
#gausfiltered = pd.DataFrame(index=sigmas, dtype='float64')

for sigma in sigmas:
    gausfiltered.append( gaussian(img_lab[:,:,0], sigma))
#imgg = gaussian(img_lab[:,:,0], 100)

'''

#%% calculating sigmas
sigmas = [2, 5, 10, 20, 50, 100] # sigmas i want to get

s2 = lambda s1, s3 : math.sqrt(s3**2 - s1**2)
working_sigmas = [sigmas[0]] # sigmas i have to add up
for s in sigmas[1:]:
    last = working_sigmas[-1]
    working_sigmas.append(s2(last, s))


#%% Gaus Filter new

gausfiltered = []

img_filtered = img_lab[:,:,0]

for s in working_sigmas:
    img_filtered = gaussian(img_filtered, s)
    gausfiltered.append(img_filtered)

#%% plot filtered

fig, axs = plt.subplots(len(gausfiltered), 1, figsize=(1.3, 8), dpi=300)

for ax, gaus, sigma in zip(axs, gausfiltered, sigmas):
    ax.imshow(gaus, cmap='Greys_r')
    ax.set_title(str(sigma))
    ax.axis('off')


#%% differences

differences = np.diff(np.array(gausfiltered), n=1, axis=0)


#%% plot differences

fig, axs = plt.subplots(len(differences), 1, figsize=(1.3, 6), dpi=300)

for ax, diff in zip(axs, differences):
    ax.imshow(diff, cmap='Greys_r')
    ax.axis('off')

#%% histogram of oriented gradients

fd, hog_image = hog(differences[0], orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)


#%% plot hog

fig, ax = plt.subplots(figsize=(4, 5), dpi=300)


ax.axis('off')
ax.imshow(hog_image, cmap=plt.cm.gray)
ax.set_title('Histogram of Oriented Gradients')
#plt.show()

