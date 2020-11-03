# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:29:16 2020

@author: Nils
"""

#import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from datetime import timedelta
#import time as tm

def plot_vector(vector, name, parameter, time=0):
    
    title = name
    title = title.replace('Google_Art_Project.jpg', '')
    title = title.replace('_-_', ' ')
    title = title.replace('_', ' ')
    
    
    microseconds = timedelta(seconds=time).microseconds
    time_text = timedelta(seconds=time) - timedelta(microseconds=microseconds)
    
    text = ''
    for key, value in parameter.items():
        text += str(key).replace('_', ' ') + ' = ' + str(value) + '\n'
    
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.text(0.15, 0.65, text, ha='left')
    fig.text(0.85, 0.8, time_text, ha='right')
    
    ax.plot(vector, '.', markersize=0.5)
    ax.set_ylim(0, 1)
    #ax.set_facecolor('Black')
    ax.set_title(title)
    
    plt.show()
    
    
def plot_img(img):
    
    fig, ax = plt.subplots(dpi = 200)
    ax.imshow(img)
    ax.axis('off')
    

def plot_lab(lab):
    
    redgreen   = LinearSegmentedColormap.from_list('a', ['green', 'white','red'])
    yellowblue = LinearSegmentedColormap.from_list('b', ['blue','white', 'yellow'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 7), dpi=200)
    
    ax1.imshow(lab[0], cmap="Greys_r", vmin=0, vmax=100)
    ax1.set_title("L channel")
    ax1.axis('off')
    
    ax2.imshow(lab[1], cmap=redgreen, vmin=-128, vmax=127)
    ax2.set_title("a channel")
    ax2.axis('off')
    
    ax3.imshow(lab[2], cmap=yellowblue, vmin=-128, vmax=127)
    ax3.set_title("b channel")
    ax3.axis('off')
    
def plot_scalespace(scalespace, labels=None):
    
    fig, axs = plt.subplots(1, len(scalespace), figsize=(10, 7), dpi=200)
    
    for index, ax in enumerate(axs):
        ax.imshow(scalespace[index], cmap="Greys_r")
        if labels:
            ax.set_title(str(labels[index]))
        else:
            ax.set_title(str(index))
        ax.axis('off')