# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:29:16 2020

@author: Nils
"""

#import matplotlib as mpl

import matplotlib

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator)

from itertools import accumulate
import numpy as np


from datetime import timedelta
#import time as tm


redgreen   = LinearSegmentedColormap.from_list('a', ['green', 'white','red'])
yellowblue = LinearSegmentedColormap.from_list('b', ['blue','white', 'yellow'])
orientation_cmap = matplotlib.cm.get_cmap('hsv')



def testplot(vector):



    x = [i for i in range(vector.size)]
    y = vector


    colors = [orientation_cmap(i) for i in np.arange(0, 1, 1/8)] * (vector.size // 8)



    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    ax.scatter(x, y, marker='.', s=0.5, c=colors)
    
    ax.set_ylim(0)
    ax.set_xlim(0, vector.size)
    

def plot_vector(vector, name, parameter, time=None):

    title = name
    title = title.replace('Google_Art_Project.jpg', '')
    title = title.replace('_-_', ' ')
    title = title.replace('_', ' ')


    parameter_text = ''
    for key, value in parameter.items():
        parameter_text += str(key).replace('_', ' ') + ' = ' + str(value) + '\n'

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.text(0.15, 0.65, parameter_text, ha='left')

    if time:
        microseconds = timedelta(seconds=time).microseconds
        time_text = timedelta(seconds=time) - timedelta(microseconds=microseconds)
        fig.text(0.85, 0.8, time_text, ha='right')


    ax.plot(vector, '.', markersize=0.5)
    ax.set_ylim(0)
    ax.set_xlim(0, vector.size)
    ax.set_title(title)


    ax.xaxis.set_major_locator(MultipleLocator(vector.size/3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.xaxis.set_minor_locator(MultipleLocator(
        vector.size / (3 * (parameter['gauß_depth']-1))
        ))

    plt.grid(b=True, which='major', axis='x', color='#666666', linestyle='-')

    plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='-', alpha=0.2)

    plt.show()


def plot_img(img):

    fig, ax = plt.subplots(dpi = 200)
    ax.imshow(img)
    ax.axis('off')


def plot_lab(lab):



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


def plot_scalespaces(scalespaces, labels=None, normalize=False):

    ss = [im for scalespace in scalespaces for im in scalespace]

    fig, axs = plt.subplots(3, len(ss)//3, figsize=(10, 4), dpi=200)

    cmaps = ['Greys_r', redgreen, yellowblue]
    vranges = [(0, 100), (-128, 127), (-128, 127)]
    vranges_n = [(-50, 50), (-128, 127), (-128, 127)]

    c_range = vranges_n if normalize else vranges

    for ix, collumn in enumerate(axs):
        for iy, ax in enumerate(collumn):


            ax.imshow(scalespaces[ix][iy],
                      cmap='Greys_r',
                      # cmap=cmaps[ix],
                       vmin=c_range[ix][0],
                       vmax=c_range[ix][1],
                      )
            if labels:
                ax.set_title(str(labels[ix]))
            else:
                ax.set_title(str([ix, iy]))
            ax.axis('off')


def plot_channel_histogramms(feature_vector, diffs, parameter, channel=[0, 1, 2]):

    fig, axs = plt.subplots(
        nrows=parameter['gauß_depth']-1,
        ncols=2,
        figsize=(15, 10),
        dpi=200,
        sharex='col',
        squeeze=True,
        # sharey='all',
        # constrained_layout=True,
        )



    l, a, b = np.split(feature_vector, 3)

    histograms = np.split(l, parameter['gauß_depth']-1)
    plt.subplots_adjust(hspace = 0, wspace= 0,  left=0)

    for index, axss, in enumerate(axs):

        ax = axss[1]
        ix = axss[0]

        names = np.arange(histograms[index].size)



        colors = [orientation_cmap(i) for i in np.arange(0, 1, 1/parameter['hist_orientations'])]

        ax.bar(names, histograms[index], color=colors, width=1, align='edge')
        # ax.set_ylim(0, 0.5)
        ax.set_xlim(0, histograms[index].size)


        # setup grid lines
        majorgrid = list(accumulate([(4**i)*parameter['hist_orientations'] for i in range(parameter['phog_depth'])]))
        ax.grid(b=True, which='major', axis='x', color='#333333', linestyle='-', linewidth=1, alpha=1)
        ax.xaxis.set_major_locator(FixedLocator(majorgrid))
        ax.grid(b=True, which='minor', axis='x', color='#666666', linestyle='-', alpha=0.4)
        ax.xaxis.set_minor_locator(MultipleLocator(parameter['hist_orientations']))
        ax.set_axisbelow(True)


        ix.imshow(diffs[0][index], cmap="Greys_r")
        ix.axis('off')
        # ax.axis('off')
        ix.set_aspect('equal')
        ax.set_aspect('auto')
        # ax.set_aspect(20)
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))


        # ax.xaxis.set_minor_locator(MultipleLocator(
        #     vector.size / (3 * (parameter['gauß_depth']))
        # ))