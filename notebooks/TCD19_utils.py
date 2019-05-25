import squarify
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba

from umap import UMAP
from math import pi
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


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


def ecdf(x, plot = None, label = None, c = None, alpha = None):
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

    if label is not None and plot is True and c is not None and alpha is not None: 
        
        plt.scatter(x_sorted, ecdf, alpha = alpha, label = label, color = c)

    elif label is not None and plot is True and c is not None: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7, label = label, color = c)
    
    elif label is not None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7, label = label)
        
    return x_sorted, ecdf


def make_treemap(x_keys, x_counts):
    
    '''
    
    Wrapper function to plot treemap using the squarify module. 
    
    -------------------------------------------
    x_keys = names of the different categories 
    x_counts = counts of the given categories
    
    '''
    
    norm = mpl.colors.Normalize(vmin=min(x_counts), vmax=max(x_counts))
    colors = [mpl.cm.Greens(norm(value)) for value in x_counts]
    
    plt.figure(figsize=(14,8))
    squarify.plot(label= x_keys, sizes= x_counts, color = colors, alpha=.6)
    plt.axis('off');



def make_radar_chart(x_keys, x_counts):
    
    '''
    Wrapper function to make radar chart.
    
    ------------------------------------------
    
    x_keys = names of the different categories 
    x_counts = counts of the given categories    
    
    '''
    
    categories = list(x_keys)
    N = len(categories)
    
    if N > 30: 
        
        print('The categories are too big to visualize in a treemap.')
        
    else:    

        values = list(x_counts)
        values.append(values[0])
        values_sum = np.sum(values[:-1])

        percentages= [(val/values_sum)*100 for val in values]

        #angles
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        sns.set_style('whitegrid')

        # Initialize figure
        plt.figure(1, figsize=(7, 7))

        # Initialise the polar plot
        ax = plt.subplot(111, polar=True)

        # Draw one ax per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=12)

        #Set first variable to the vertical axis 
        ax.set_theta_offset(pi / 2)

        #Set clockwise rotation
        ax.set_theta_direction(-1)

        #Set yticks to gray color 

        ytick_1, ytick_2, ytick_3 = np.round(max(percentages)/3),np.round((max(percentages)/3)*2),np.round(max(percentages)/3)*3

        plt.yticks([ytick_1, ytick_2, ytick_3], [ytick_1, ytick_2, ytick_3],
                   color="grey", size=10)

        plt.ylim(0, int(max(percentages)) + 4)


        # Plot data
        ax.plot(angles, percentages, linewidth=1,color = 'lightgreen')

        # Fill area
        ax.fill(angles, percentages, 'lightgreen', alpha=0.3);    




def palette(cmap = None):

	palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15)
	

	if cmap == True:
		palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15, as_cmap = True)

	return palette 


def convert_to_datetime(df, col):
	    
    col = pd.to_datetime(col)
    
    return col


def AVIS_analysis(path):


	cols = ['PartnerDCMX', 'Distribution_Channel_Details','Reservationid',
	        'Reservation_Country_Code','Reservation','ChkOut_Date',
	        'ChkOut_Time', 'ChkIn_Date', 'Booking_Date', 'Booking_Time',
	        'Cancelled_Date', 'ChkOut_Sta_Num', 'ChkIn_Sta_Num', 'Booking_Sta_Num',
	        'Quoted_Rate_Code', 'Conf_Veh_Class_Code', 'AWD','Price_Quote_Amt_Local',
	        'Price_Quote_Amt_USD','ATC_Iata_Num','Travel_Agency_Name', 'Res_Status',
	        'Country_Of_Origin_Code', 'Distribution_Channel','Ra_Number', 'transaction_date',
	        'Origen', 'Leadtime','Leadtime_Rangos', 'LOR', 'LOR_Rangos','Rate_Type',
	        'Rate_Segment', 'Dist_Channel', 'BK_Weekday','CO_Weekday', 'CO_DateTime', 'Booking_DateTime']

	df = pd.read_csv(path, names = cols) 


	df.ChkIn_Date = convert_to_datetime(df, df.ChkIn_Date)
	df.ChkOut_Date = convert_to_datetime(df, df.ChkOut_Date)
	df.ChkOut_Time = convert_to_datetime(df, df.ChkOut_Time)
	df.Booking_Date = convert_to_datetime(df, df.Booking_Date)
	df.transaction_date = convert_to_datetime(df, df.transaction_date)
	df.CO_DateTime = convert_to_datetime(df, df.CO_DateTime)
	df.Booking_DateTime = convert_to_datetime(df, df.Booking_DateTime)


	gb = df.groupby('Booking_Date').size()

	#print(type(gb.index))

	by_weekday = gb.groupby(gb.index.dayofweek).mean()


	fig, ax = plt.subplots(figsize = (7,4))

	ax.plot(by_weekday.values, color = 'dodgerblue', lw = 4)
	ax.set_xticklabels(['','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
	ax.set_xlabel('day of week')
	ax.set_ylabel('$bookings$')

	ax.set_title('Mean number of bookings by day of week')

	#plt.savefig('../Desktop/avis_plots/weekly_booking_patterns.png',
	#			 dpi = 420, bbox_inches = 'tight');



	gb_dt = df.groupby('Booking_DateTime').size()

	hourly = gb_dt.resample('h').sum()

	df_hourly = pd.DataFrame(hourly, columns = ['counts'])

	gb_hourly_mean = hourly.groupby(hourly.index.time).mean()

	fig, ax = plt.subplots()

	gb_hourly_mean.plot(color = 'thistle')

	ax.set_ylabel('$\mu_{bookings}$')
	ax.set_title('Daily booking patterns')
	plt.tight_layout()
	#plt.savefig('../Desktop/avis_plots/daily_booking_patterns.png',
	#			 dpi = 420, bbox_inches = 'tight')

	pivoted = df_hourly.pivot_table('counts', index = df_hourly.index.time,
	                            columns = df_hourly.index.date)

	pivoted.plot(legend = False, alpha = 0.15, figsize = (7,4))
	plt.ylabel('$bookings$')

	plt.tight_layout()
	#plt.savefig('../Desktop/avis_plots/hourly_booking_phantom.png',
	#			 dpi = 420, bbox_inches = 'tight')


	df_pivoted_T = pivoted.T

	x = df_pivoted_T.fillna(0).values

	total_bookings = x.sum(1)


	reducer = UMAP().fit_transform(x)

	plt.figure(figsize = (8,5))
	plt.scatter(reducer[:,0], reducer[:,1], alpha = 0.5, color = 'dodgerblue')

	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	plt.tight_layout()
	#plt.savefig('../Desktop/avis_plots/UMAP_first_look.png',
	#			 dpi = 420, bbox_inches = 'tight')


	plt.figure(figsize = (8,5))
	plt.scatter(reducer[:,0], reducer[:,1], alpha = 0.9, c = total_bookings, 
	           cmap = 'magma_r')

	plt.colorbar(label='total bookings per day');


	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')

	#plt.savefig('../Desktop/avis_plots/UMAP_colored_bookings_per_day.png',
	#			 dpi = 420, bbox_inches = 'tight');


	gmm = GMM(n_components=2).fit(reducer)
	labels = gmm.predict(reducer)

	plt.figure(figsize = (8,5))

	plt.scatter(reducer[:,0], reducer[:,1], alpha = 0.7, c = labels)

	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	plt.tight_layout()
	#plt.savefig('../Desktop/avis_plots/UMAP_colored_2_clusters.png',
	#			 dpi = 420, bbox_inches = 'tight');

	dayofweek = pd.DatetimeIndex(pivoted.columns).dayofweek

	classes = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
	      'Saturday', 'Sunday']

	plt.figure(figsize = (8,5))

	plt.scatter(reducer[:, 0], reducer[:,1], c = dayofweek, alpha = 0.8,
	            cmap = 'magma_r')

	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')

	cbar = plt.colorbar(boundaries=np.arange(8)-0.5)
	cbar.set_ticks(np.arange(7))
	cbar.set_ticklabels(classes)

	plt.tight_layout()
	#plt.savefig('../Desktop/avis_plots/umap_clustered.png', dpi = 500, bbox_inches='tight');


def col_encoding(df, column):
    
    """
    Returns a one hot encoding of a categorical colunmn of a DataFrame.
    
    ------------------------------------------------------------------
    
    inputs~~

    -df:
    -column: name of the column to be one-hot-encoded in string format.
    
    outputs~~
    
    - hot_encoded: one-hot-encoding in matrix format. 
    
    """
    
    le = LabelEncoder()
    
    label_encoded = le.fit_transform(df[column].values)
    
    hot = OneHotEncoder(sparse = False)
    
    hot_encoded = hot.fit_transform(label_encoded.reshape(len(label_encoded), 1))
    
    return hot_encoded


def one_hot_df(df, cat_col_list):
    
    """
    Make one hot encoding on categoric columns.
    
    Returns a dataframe for the categoric columns provided.
    -------------------------
    inputs
    
    - df: original input DataFrame
    - cat_col_list: list of categorical columns to encode.
    
    outputs
    - df_hot: one hot encoded subset of the original DataFrame.
    """

    df_hot = pd.DataFrame()

    for col in cat_col_list:     

        encoded_matrix = col_encoding(df, col)

        df_ = pd.DataFrame(encoded_matrix,
                           columns = [col+ ' ' + str(int(i))\
                                      for i in range(encoded_matrix.shape[1])])

        df_hot = pd.concat([df_hot, df_], axis = 1)
        
    return df_hot

@numba.jit(nopython=True)
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    """
    return np.random.choice(data, size=len(data))


def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

