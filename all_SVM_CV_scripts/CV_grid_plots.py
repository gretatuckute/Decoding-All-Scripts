# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:15:12 2019

@author: Greta
"""

# Utility function to move the midpoint of a colormap to be around
# the values of interest.
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
    

def plotHeatmap(inputfile,scores='test',title='Heatmap'):
    C_vals, gamma_vals, train_scores, test_scores = getVals2(inputfile)
    
    if scores == 'test':
        reshape_scores = np.reshape(test_scores,[5,5]) # Input the accuracies in a list, and reshape to e.g. 5x5
        
    if scores == 'train':
        reshape_scores = np.reshape(train_scores,[5,5])
        
    random_chance = 0.5
    C_range = C_vals[0:5] # Add list with C vals here
    gamma_range = gamma_vals[0::5] # Add list with gamma vals here

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(reshape_scores, interpolation='nearest', cmap=plt.cm.hot, # evt. cmap=plt.cm.RdBu_r 
               norm=MidpointNormalize(vmin=random_chance - 0.1, midpoint=random_chance),vmax=random_chance+0.1)
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.colorbar()
    plt.yticks(np.arange(len(gamma_range)), gamma_range)
    plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
    plt.title(title)
    plt.show()
    
    return plt


