import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 

<<<<<<< HEAD

def set_plotting_style():
      
    tw = 1.5
    rc = {'lines.linewidth': 2,
            'axes.labelsize': 18,
            'axes.titlesize': 21,
            'xtick.major' : 16,
            'ytick.major' : 16,
=======
def set_plotting_style():
      
      tw = 1.5
      rc = {'lines.linewidth': 2,
            'axes.labelsize': 22,
            'axes.titlesize': 24,
            'xtick.major' : 22,
            'ytick.major' : 22,
>>>>>>> 3e84c208b10f4da1038faf13380ff13fe2ed787f
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
<<<<<<< HEAD
            'legend.fontsize': 12, 
           "xtick.direction": "in","ytick.direction": "in"}
    
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)
    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)
=======
            'legend.fontsize': 22, 
           "xtick.direction": "in","ytick.direction": "in"}
>>>>>>> 3e84c208b10f4da1038faf13380ff13fe2ed787f


def set_plotting_style_2():
      
<<<<<<< HEAD
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
=======
      tw = 1.5
      rc = {'lines.linewidth': 2,
            'axes.labelsize': 22,
            'axes.titlesize': 24,
            'xtick.major' : 22,
            'ytick.major' : 22,
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
            'legend.fontsize': 22, 
           "xtick.direction": "in","ytick.direction": "in"}



      plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
      plt.rc('mathtext', fontset='stixsans', sf='sans')
      sns.set_style('ticks', rc=rc)

      sns.set_palette("colorblind", color_codes=True)
      sns.set_context('notebook', rc=rc)


>>>>>>> 3e84c208b10f4da1038faf13380ff13fe2ed787f
