import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# Importing everything from above
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from scipy.signal import correlate
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import gridspec
import pywt

# define colors
colors = {'GRAY1':'#231F20', 'GRAY2':'#414040', 'GRAY3':'#555655',
         'GRAY4':'#646369', 'GRAY5':'#76787B', 'GRAY6':'#828282',
         'GRAY7':'#929497', 'GRAY8':'#A6A6A5', 'GRAY9':'#BFBEBE',
         'BLUE1':'#174A7E', 'BLUE2':'#4A81BF', 'BLUE3':'#94B2D7',
         'BLUE4':'#94AFC5', 'RED1':'#C3514E', 'RED2':'#E6BAB7',
         'GREEN1':'#0C8040', 'GREEN2':'#9ABB59', 'ORANGE1':'#F79747',}

# defining some constants
column_names=['Timestamp','Module','Type','Temp_Mod', 'VBus',
              'PT100(0)', 'PT100(1)', 'LVL_Dim(1)', 'V_MPPT', 
              'V_Panel','LVL_Drain(1)','VBat', 'V_Supp','Temp_Oil',
              'Temp_gab','V_MPPT_TE','V_Panel_TE']

def extract_from_date(data, timestamp_column, suffix):
    '''
    Add time fetures to dataset
    '''
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    
    data[str(suffix + '_year')] = data[timestamp_column].dt.year
    data[str(suffix + '_month')] = data[timestamp_column].dt.month
    data[str(suffix + '_month_name')] = data[timestamp_column].dt.month_name()
    data[str(suffix + '_year_month')] = data[timestamp_column].dt.strftime('%Y%m')
    data[str(suffix + '_date')] = data[timestamp_column].dt.strftime('%Y%m%d')
    data[str(suffix + '_week')] = data[timestamp_column].dt.isocalendar().week
    data[str(suffix + '_day')] = data[timestamp_column].dt.day
    data[str(suffix + '_dayofweek')] = data[timestamp_column].dt.dayofweek
    data[str(suffix + '_day_name')] = data[timestamp_column].dt.day_name()
    data[str(suffix + '_hour')] = data[timestamp_column].dt.hour
    data[str(suffix + '_day_time')] = pd.cut(data[str(suffix + '_hour')], bins=[0, 12, 18, 23], labels=['morning', 'afternoon', 'night'])
    data[str(suffix + '_weekend')] = ((data[str(suffix + '_dayofweek')] >= 5) & (data[str(suffix + '_dayofweek')] <= 6)).astype(int)
    
    return data

def load_powers(path_of_file):
    
    columns_radio2 = ['Timestamp','Module','Type', 'Receiver', 'Tx1', 'P_Tx1(dbm)',
                      'Tx2', 'P_Tx2(dbm)', 'Tx3', 'P_Tx3(dbm)',
                      'Tx4','P_Tx4(dbm)', 'Tx5', 'P_Tx5(dbm)',
                      'Tx6', 'P_Tx6(dbm)', 'Tx7','P_Tx7(dbm)']
    
    # Loading the data
    df_powers = pd.read_csv(path_of_file, usecols=columns_radio2, parse_dates=['Timestamp'],                            
                            dtype={'P_Tx1(dbm)':float, 'P_Tx2(dbm)':float, 'P_Tx3(dbm)':float,
                                   'P_Tx4(dbm)':float, 'P_Tx5(dbm)':float, 'P_Tx6(dbm)':float,
                                   'Tx1':str, 'Tx2':str, 'Tx3':str, 'Tx4':str, 'Tx5':str, 'Tx6':str,
                                   'Tx7':str, 'P_Tx7(dbm)':float})

    return df_powers

def annotate_percentage(ax, data=None, total = None,title=None,
                        horizontal=False, size=14, fontsize=18, yy=1.02, offset=20):
    '''
    Annotates a percentage and the amounth at the top of the bar plot.
    Sets the title of the plot
    yy: height of the title
    fontsize: fontsize of the title
    size: is the size of the annotation
    '''
    if horizontal:
        if total is None:
            total = float(len(data))
        xmax=ax.get_xlim()[1] 
        offset = xmax*0.005
        for p in ax.patches:
            text = '{:.1f}'.format(p.get_width())
            x = p.get_x() + p.get_width() + offset
            y = p.get_height()/2 + p.get_y()
            ax.annotate(text, (x, y), size=size)
        ax.set_title(title, fontsize=fontsize, y=yy)
    else:
        if total is None:
            total = float(len(data))
        xmax=ax.get_xlim()[1] 
        offset = xmax*0.005
        for p in ax.patches:
            percentage = '{:.1f}%\n{:.1f}'.format(100 * p.get_height()/total, p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_height() + offset
            ax.annotate(percentage, (x, y), ha='center', size=size)
        ax.set_title(title, fontsize=fontsize, y=yy)
    
def get_iqr(df, feature, k_factor = 1.5, remove=True):
    '''
    Return the interquartile range and defines an outlier range
    based in a k factor.
    if remove==True: returns a dataframe with the removed outliers
    and a dataframe with the outliers
    '''
    q25, q75 = np.percentile(df[feature], 25), np.percentile(df[feature], 75)
    IQR = q75 - q25
    cut_off = IQR * k_factor
    lower, upper = q25 - cut_off, q75 + cut_off
    if remove:
        data = df.loc[(df[feature] > lower) & (df[feature] < upper)]
        outliers = df.loc[(df[feature] < lower) & (df[feature] > upper)]
        return data, outliers
    return IQR, lower, upper

def annotate_total(ax, data=None, horizontal=False, total = None,
                   title=None, size=14, fontsize=18, yy=1.02, offset=20):
    '''
    Annotates a percentage and the amounth at the top of the bar plot.
    Sets the title of the plot
    yy: height of the title
    fontsize: fontsize of the title
    size: is the size of the annotation
    horizontal: defines if the bars are horizontal or not
    '''

    if horizontal:
        if total is None:
            total = float(len(data))
        xmax=ax.get_xlim()[1] 
        offset = xmax*0.005
        for p in ax.patches:
            text = '{:.1f}'.format(p.get_width())
            x = p.get_x() + p.get_width() + offset
            y = p.get_height()/2 + p.get_y()
            ax.annotate(text, (x, y), size=size)
        ax.set_title(title, fontsize=fontsize, y=yy)
    else:
        if total is None:
            total = float(len(data))
        xmax=ax.get_xlim()[1] 
        offset = xmax*0.005
        for p in ax.patches:
            text = '{:.1f}'.format(p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_height() + offset
            ax.annotate(text, (x, y), ha='center', size=size)
        ax.set_title(title, fontsize=fontsize, y=yy)
    
def remove_chart_borders(ax):
    # remove chart border
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    
def annotate(ax, text, x=0.5, y=0.5, line=False,stacked=False, color='#94AFC5', fontsize=16, linespacing=1.45):
    '''
    Add text annotation in plot using x and y in percents.
    final y value is 100, must specify if the graph is stacked,
    this is to allow insert text at the end of the figure.
    '''
    if stacked:
        X_end = (len(ax.patches)/2)
    else:
        X_end = (len(ax.patches))
    if line:
        X_end = ax.axes.get_xlim()[1]
    y_end = ax.axes.get_ylim()[1]
    
    plt.text(X_end*x, y_end*y, text, 
             fontsize=fontsize, linespacing=linespacing, 
             color=color)
    
def line_plot_annotate(ax, values, x, y, fontsize=14, timestamp=False):
    '''
    values is the array of number we want to annotate
    x is the values in the x axis, can be an index value
    from a pandas dataframe.
    y is the values in the y axis, may be the same as values
    if the annotations are the values
    Timestamp: True converts index to timestamp
    
    Example:
    values = df.loc[state].sort_index()[:'201803']['order_id'].values
    x = df.loc[state].sort_index()[:'201803']['order_id'].index
    y = df.loc[state].sort_index()[:'201803']['order_id'].values
    '''
    if timestamp:
        for i, txt in enumerate(values):
            ax.annotate(txt, (Timestamp(x[i]), y[i]),fontsize=fontsize)
    else:
        for i, txt in enumerate(values):
            ax.annotate(txt, (x[i], y[i]),fontsize=fontsize)
        
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

def plotMovingAverage(series, window=4, plot_intervals=False, scale=1.96, plot_anomalies=False, figsize=(20,5)):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=figsize)
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "darkgreen", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=8)

    plt.plot(series[window:], label="Real values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
    
def plotExponentialSmoothing(series, alphas, figsize=(20,5)):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=figsize)
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Real")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);
        
def get_module_feature(module, features, path_of_file, numerical=False):
    
    df = pd.read_csv(path_of_file, parse_dates=['Timestamp'], index_col='Timestamp')
    
    try:
        df = df.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    if numerical:
        return np.array(df[df['Module']==module][[features]])
    else:
        df[df['Module']==module][[features]]
        

def cross_correlation(signal1, signal2):
    '''
    Calculate cross correlation between two time series
    The 2 signals must be sampled at same freq 
    returns: lags, ccov, ccor
    
    Example:
    signal1 = data['29.E5.5A.24(Sensor)'][:1000].dropna()
    signal2 = data['PT100(Sensor)'][:1000].dropna()
    lags, ccov, ccor = cross_correlation(signal1, signal2)

    '''
    # Make them same size by dropping the final NAN values
    
    signal1 = signal1.dropna().values
    signal2 = signal2[:len(signal1)].values
    
    # create array of lags
    lags = np.arange(-len(signal1) + 1, len(signal1))
    
    ccov = correlate(signal1 - signal1.mean(),
                            signal2 - signal2.mean(),
                            mode='full',
                            method='auto')
    ccor = ccov / (len(signal1) * signal1.std() * signal2.std())
    
    return lags, ccov, ccor
        
    
def plot_ccross(signal1, signal2, lags, ccov, ccor, ax):
    '''
    Example:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,7))
    ax1.plot(np.arange(0, len(signal1)), signal1, 'b', label='Signal1')
    ax1.plot(np.arange(0, len(signal2)), signal2, 'r', label='Signal2')
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    plot_ccross(signal1, signal2, lags, ccov, ccor, ax2)
    plt.tight_layout()
    '''
    ax.plot(lags, ccor)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('Cross-correlation')
    ax.set_xlabel('lag of signal 1 relative to signal 2')
    ax.set_title('Max cross-correlation of {} in lag {}'.format(ccor.max(), lags[np.argmax(ccor)]), y=1.02)
    
def plot_wavelet(signal, period_min=10, max_scale= 100, wavelet='morl', cmap='twilight_shifted',
                 alpha=0.4, linewidth=2, plot_over=True, title=None, figsize=(24,6), colorbar=True):
    '''
    Plot wavelet transform of a signal.
    You must set a constat frequency before. This can be done by resampling
    period_min: Int: sampling period in minutes.
    max_scale: Int: Uppert limit for scales
    
    return ax
    
    Example:
    
    init_date = '2019-01-01'
    end_date = '2019-01-14'
    module1 = '29.E5.5A.24(Sensor)'
    module2 = '00.57.FE.0E(R5)'
    # take a week
    signal = data[module1][init_date:end_date].resample('10Min').mean().interpolate()

    ax1 = plot_wavelet(signal_1, max_scale=180, wavelet='morl', cmap='coolwarm',
                   title=('CWT of {} from {} to {} using Morlet').format(module1, init_date, end_date))

    '''
    
    scales = np.arange(1, max_scale)

    # Separete Dates
    fig, ax = plt.subplots(1, 1, figsize=(24,6), sharex=True, sharey=True)
    coeffs, freqs = pywt.cwt(signal.values, scales=scales, wavelet=wavelet)
    # create scalogram
    pcm = ax.imshow(coeffs, cmap = cmap, aspect = 'auto')
    ax.set_ylabel('Periods')
    ax.set_xlabel('Time (Min)')
    ax.set_title(title)
    
    one_day = (24*60)/period_min # 24 hours * 60 minutes / 10 minutes of resample time
    
    ax.set_xticks(np.arange(0, len(signal), one_day))
    
    if plot_over:
        ax2 = ax.twinx()
        ax2.plot(signal.values, color ='k', alpha=alpha, linewidth=linewidth)
        ax2.set_yticks([])
        if colorbar:
            fig.colorbar(pcm, ax=ax2)
    else:
        if colorbar:
            fig.colorbar(pcm, ax=ax1)
    
    for ii in range(len(signal)):
        if ii % one_day == 0:   # a day with resample time = 10 min
            ax.axvline(x=ii, color=colors['GRAY1'], linestyle = '--')
    return ax
    
def wavelet_transform(signal, scales=np.arange(1,100), wavelet='morl', cmap='coolwarm'):

    # Separete Dates
    coeffs, freqs = pywt.cwt(signal1.values, scales=scales, wavelet=wavelet)
    # create scalogram
    return coeffs, freqs

def check_symmetry(ax, dff, rec_1, rec_2, tx1, tx2, plot=True, hist=False):
    '''
    Plots the data that receiver 1 gets from receiver 2
    and viceversa to see the difference between them.
    What radio 1 sees from radio 2 and what radio 2
    sees from radio 1.
    
    Example:
    check_symmetry(dff, '00.57.FE.04', '00.57.FE.05', '0x0057FE05', '0x0057FE04')
    tx1 is rec_2 in hexadecimal format (0x00...)
    
    Example to see symmetry between all modules:
    
    transmitter = '0x0057FE0F'
    rec_2 = '00.57.FE.0F'
    fig, axx = plt.subplots(4, 2, figsize=(24, 20))
    plt.delaxes(ax=axx[3,1])
    for rec, hex_rec, axi in zip(dff.Receiver.unique(), Receivers, axx.flat):
        if rec != '00.57.FE.0F' and hex_rec != transmitter:
            mean_sym, reciprocal = check_symmetry(axi, dff, rec, rec_2, transmitter, hex_rec, plot=True)
            #plt.setp(axi.get_xticklabels(), rotation=30, ha='right')
    axi.set_yticks(np.arange(-20, -95, -10))
    '''
    
    received = dff[dff['Receiver'] == rec_1].resample('12H').mean()[tx1]
    transmitted = dff[dff['Receiver'] == rec_2].resample('12H').mean()[tx2]
    reciprocal = pd.concat([received, transmitted], axis=1)
    rolling_mean = reciprocal.rolling(window=12).mean()
    
    reciprocal['diffe'] = (np.abs(reciprocal[tx1] - reciprocal[tx2]) * -1).round(2)
    mean_sym = (np.abs(reciprocal[tx1] - reciprocal[tx2]) * -1).mean()
    
    if plot:
        if hist:
            sns.histplot(x = tx1, data=reciprocal, kde=False, 
                         label=(f'{rec_1} sees from {tx1}'),
                         color=colors['BLUE2'], ax=ax)
            sns.histplot(x = tx2, data=reciprocal, kde=False, 
                         label=(f'{rec_1} sees from {tx2}'),
                         color=colors['RED1'], ax=ax)
        else:
            ax.plot(reciprocal.index, reciprocal[tx1], label=(f'{rec_1} sees from {tx1}'))
            ax.plot(reciprocal.index, reciprocal[tx2], label=(f'{rec_2} sees from {tx2}'))
            ax.plot(rolling_mean.index, rolling_mean[tx1], color=colors['GRAY3'])
            ax.plot(rolling_mean.index, rolling_mean[tx2], color=colors['GRAY3'])
            
        ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0.6, 0.5, 0.5))    
        plt.tight_layout()
        return mean_sym, reciprocal 
    
def isolation_forest(data, scaler='MinMaxScaler', contamination=0.01, n_estimators=50):
    '''
    Runs an isolation forest in the dataset and detects the outliers
    data is a pandas Series
    
    return anomaly_Iforest, data
    '''
    df = data.copy()
    col = data.columns[0]
    index = data.index
    df = df.reset_index(drop=True)
    
    if scaler=='MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    train = pd.DataFrame(scaler.fit_transform(df), columns=[df.columns])
    model =  IsolationForest(contamination=0.01, n_estimators=50, max_samples='auto')
    model.fit(train)

    # add the data to the main
    df['scores_isolation_f'] = pd.Series(model.decision_function(train))
    df['anomaly_IsolationF'] = pd.Series(model.predict(train))
    df['anomaly_IsolationF'] = df['anomaly_IsolationF'].map( {1: 0, -1: 1} )

    df = df.set_index(index, drop=True)
    # anomalies marked as 1
    anomaly_Iforest = df.loc[df['anomaly_IsolationF'] == 1, [col]] #anomaly
    print(anomaly_Iforest.describe().transpose())
    
    return anomaly_Iforest, df

def oneClass_SVM(data, nu=0.95, outliers_fraction=0.1, scaler='MinMaxScaler'):
    '''
    Runs one class svm algorithms in the dataset and detects the outliers
    data is a pandas Series
    
    return anomaly_svm, df
    '''
    df = data.copy()
    col = data.columns[0]
    index = data.index
    df = df.reset_index(drop=True)
    
    if scaler=='MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    train = pd.DataFrame(scaler.fit_transform(df), columns=[df.columns])
    model =  OneClassSVM(nu=nu * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
    model.fit(train)

    # add the data to the main  
    df['anomaly_SVM'] = pd.Series(model.predict(train))
    df['anomaly_SVM'] = df['anomaly_SVM'].map( {1: 0, -1: 1} )
    
    df = df.set_index(index, drop=True)
    # anomalies marked as 1
    anomaly_svm = df.loc[df['anomaly_SVM'] == 1, [col]] #anomaly
    
    return anomaly_svm, df

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_anomaly(data, anomalies, column, ax, plot_hist=False):
    '''
    Plot distribution of data and distribution of anomalies.
    Plot the anomalies as points above the data
    '''
    if plot_hist:
        fig = plt.figure(figsize=(24,5))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 3])
        ax1 = fig.add_subplot(spec[0])
        sns.histplot(data = data, x=column, ax=ax1, color=colors['BLUE2'], bins=100)
        sns.histplot(data = anomalies, x = column, kde=False, color=colors['RED1'], ax=ax1, bins=20)
        ax2 = fig.add_subplot(spec[1])
        sns.lineplot(data = data, x = data.index, y=data[column], ax=ax2, color=colors['BLUE2'])
        sns.scatterplot(data = anomalies, x=anomalies.index, y=anomalies[column], color=colors['RED1'], s = 50)
        plt.delaxis(ax)
        plt.tight_layout()
    else:
        sns.lineplot(data = data, x = data.index, y=data[column], color=colors['BLUE2'], ax=ax,)
        sns.scatterplot(data = anomalies, x=anomalies.index, y=anomalies[column], color=colors['RED1'], s = 50, ax=ax)
        ax.set_title(f'Anomalies in data from module {column}')
        plt.tight_layout()