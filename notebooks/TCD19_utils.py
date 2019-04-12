import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 


def set_plotting_style():
      
    tw = 1.5
    rc = {'lines.linewidth': 2,
            'axes.labelsize': 18,
            'axes.titlesize': 21,
            'xtick.major' : 16,
            'ytick.major' : 16,
            'xtick.major.width': tw,
            'xtick.minor.width': tw,
            'ytick.major.width': tw,
            'ytick.minor.width': tw,
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',
            'font.family': 'serif',
            'weight':'bold',
            'grid.linestyle': ':',
            'grid.linewidth': 1.5,
            'grid.color': '#ffffff',
            'mathtext.fontset': 'stixsans',
            'mathtext.sf': 'fantasy',
            'legend.frameon': True,
            'legend.fontsize': 12, 
           "xtick.direction": "in","ytick.direction": "in"}
    
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)
    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)


def set_plotting_style_2():
      
    tw = 1.5

    rc = {'lines.linewidth': 2,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.major' : 16,
        'ytick.major' : 16,
        'xtick.major.width': tw,
        'xtick.minor.width': tw,
        'ytick.major.width': tw,
        'ytick.minor.width': tw,
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'font.family': 'sans',
        'weight':'bold',
        'grid.linestyle': ':',
        'grid.linewidth': 1.5,
        'grid.color': '#ffffff',
        'mathtext.fontset': 'stixsans',
        'mathtext.sf': 'fantasy',
        'legend.frameon': True,
        'legend.fontsize': 12, 
       "xtick.direction": "in","ytick.direction": "in"}



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)

    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)


def ecdf(x, plot = None, label = None):
    '''
	Compute and plot ECDF. 

	----------------------
	Inputs

	x: array or list, distribution of a random variable
    
    plot: bool, if True return the plot of the ECDF

    label: string, label for the plot
	
	Outputs 

	x_sorted : sorted x array
	ecdf : array containing the ECDF of x


    '''
    x_sorted = np.sort(x)
    
    n = len (x)
    
    ecdf = np.arange(0, len(x_sorted)) / len(x_sorted)
    
    if label is not None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7, label = label)
        
    return x_sorted, ecdf

def palette(cmap = None):

	palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15)
	

	if cmap == True:
		palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15, as_cmap = True)

	return palette 

