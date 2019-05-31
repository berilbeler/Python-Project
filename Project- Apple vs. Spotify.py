import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

dfa = pd.read_csv("appledata.csv", sep=";")
seriesname = 'Subscribers'
apple = dfa[seriesname]
apple.replace(0, np.nan, inplace=True)
apple = apple.fillna(method = "bfill")
apple = apple.fillna(method = "ffill")
size=len(apple)
applearray = np.asarray(apple, dtype= float)


appledata=[]
i=0
while i+2 < len(applearray):
    total=int(applearray[i]+applearray[i+1]+applearray[i+2])
    avg=total/3
    appledata.append(int(avg))
    i =i+3
#print("Apple:", appledata)

growthapple=[]
i=0
while i+1 < len(appledata):
    growth= appledata[i+1]/appledata[i]-1
    growthapple.append(growth)
    i =i+1
#print("apple", growthapple)
    

dfs= pd.read_csv("spotifydata.csv", sep=";")
spotifydata = dfs[seriesname][3:len(dfs)-1]

spotifyarray = np.asarray(spotifydata, dtype= float)
#print("Spotify:", spotifyarray)

growthspotify=[]
i=0
while i+1 < len(spotifyarray):
    growth= (spotifyarray[i+1]-spotifyarray[i])/spotifyarray[i]
    growthspotify.append(growth)
    i =i+1
#print("spotify",growthspotify)

dfapple = pd.DataFrame(growthapple)
dfspotify = pd.DataFrame(growthspotify)
frames=[dfapple, dfspotify]
df_merged=pd.concat(frames,axis=1, keys=["apple","spotify"])
#print(df_merged)
print(df_merged.corr())
print("\n")
print(df_merged.cov())

########################################################

def decomp(df,f,mod='Additive'):
    result = sm.tsa.seasonal_decompose(df,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show()
    return result

decomp(applearray,f=1,mod='Additive')
decomp(spotifyarray,f=1,mod='Additive')

def test_stationarity(timeseries, color1, color2, color3):
    rolmean = pd.Series(timeseries).rolling(window=1).mean()
    rolstd = pd.Series(timeseries).rolling(window=1).std()
    orig = plt.plot(timeseries, color1,label='Original')
    mean = plt.plot(rolmean, color2, label='Rolling Mean')
    std = plt.plot(rolstd, color3, label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(applearray, color1='red', color2='purple', color3='black')
test_stationarity(spotifyarray, color1='yellow', color2='green', color3='blue')

##########################################################

#Function for Naive
def estimate_naive(dfarray):
     return float(dfarray[-1])
    
naive = round(estimate_naive (applearray),4)
print ("Naive estimation for apple:", naive)
naive = round(estimate_naive (spotifyarray),4)
print ("Naive estimation for spotify:", naive)

# Function for Simple Average
def estimate_simple_average(dfarray):
    avg = dfarray.mean()
    return avg

simpleaverage_apple = round( estimate_simple_average(applearray), 4)
print("Simple average estimation apple:", simpleaverage_apple)
simpleaverage_spotify = round( estimate_simple_average(spotifyarray), 4)
print("Simple average estimation spotify:", simpleaverage_spotify)

# Function for Moving Average
def estimate_moving_average(df,seriesname,windowsize):
    avg = df.rolling(windowsize).mean().iloc[-1]
    return avg

window = 1
movingaverage_apple = round(estimate_moving_average(apple,'Subscribers',window),4)
print("Moving average estimation for appple:", movingaverage_apple)
movingaverage_spotify = round(estimate_moving_average(spotifydata,'Subscribers',window),4)
print("Moving average estimation for spotify:", movingaverage_spotify)

# Function for Simple Exponential Smoothing
def estimate_ses(df, alpha=0.2):
    estimate = SimpleExpSmoothing(df).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate

alpha = 0.2
ses_apple = round (estimate_ses(applearray, alpha)[0], 4)
print("Exponential smoothing estimation for apple with alpha =", alpha, ": ", ses_apple)
ses_spotify = round (estimate_ses(spotifyarray, alpha)[0], 4)
print("Exponential smoothing estimation for spotify with alpha =", alpha, ": ", ses_spotify)

# Trend estimation with Holt
def estimate_holt(df, alpha=0.2, slope=0.1):
    model = Holt(df)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(1)[-1]
    return estimate

alpha = 0.2
slope = 0.1
holt_apple = round(estimate_holt(applearray,alpha, slope),4)
print("Holt trend estimation for apple with alpha =", alpha, ", and slope =", slope, ": ", holt_apple)
holt_spotify = round(estimate_holt(spotifyarray,alpha, slope),4)
print("Holt trend estimation for spotify with alpha =", alpha, ", and slope =", slope, ": ", holt_spotify)

# There is a false seasonality in apple dataset that is resulted from missing values that are filled with forward and backward
# methods. Also, because of the same reason, naive approach gives 0 error value which is not reliable.
# Therefore, we preferred not to use naive and holt winters estimation method for both of the methods since we would like to move
# with the same method which gives minimum error in total.

###################################################################

appledf = pd.DataFrame(columns=['Subscribers'])
appledf['Subscribers'] = appledata

sizeapple = len(appledf)
traina = appledf[:10]
testa = appledf[11:]
testarraya = np.asarray(testa.Subscribers)

#Simple average approach
print("Simple Average apple")
y_hat_avg = testa.copy()
y_hat_avg['avg_forecast'] = traina['Subscribers'].mean()
rms_simple_apple = sqrt(mean_squared_error(testa.Subscribers, y_hat_avg.avg_forecast))
print("RMSE Simple Average for apple: ",rms_simple_apple)

#Moving average approach
print("Moving Average apple")
windowsize = 1
y_hat_avg = testa.copy()
y_hat_avg['moving_avg_forecast'] = traina['Subscribers'].rolling(windowsize).mean().iloc[-1]
rms_moving_apple = sqrt(mean_squared_error(testa.Subscribers, y_hat_avg.moving_avg_forecast))
print("RMSE Moving Average for apple: ",rms_moving_apple)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing apple")
y_hat_avg = testa.copy()
alpha = 0.99
fit2 = SimpleExpSmoothing(np.asarray(traina['Subscribers'])).fit(smoothing_level=alpha,optimized=True)
y_hat_avg['SES'] = fit2.forecast(len(testa))
rms_ses_apple = sqrt(mean_squared_error(testa.Subscribers, y_hat_avg.SES))
print("RMSE SES for apple: ",rms_ses_apple)

# Holt
print("Holt apple")
sm.tsa.seasonal_decompose(traina.Subscribers,freq = 1).plot()
result = sm.tsa.stattools.adfuller(traina.Subscribers)
# plt.show()

y_hat_avg = testa.copy()
alpha = 0.085
fit1 = Holt(np.asarray(traina['Subscribers'])).fit(smoothing_level = alpha,smoothing_slope = 3.75)
y_hat_avg['Holt_linear'] = fit1.forecast(len(testa))
rms_holt_apple = sqrt(mean_squared_error(testa.Subscribers, y_hat_avg.Holt_linear))
print("RMSE Holt for apple: ",rms_holt_apple)

###################################################################

spotifydf = pd.DataFrame(columns=['Subscribers'])
spotifydf['Subscribers'] = spotifydata

sizespotify = len(spotifydf)
trains = spotifydf[:10]
tests = spotifydf[11:]
testarrays = np.asarray(tests.Subscribers)

#Simple average approach
print("Simple Average spotify")
y_hat_avg = tests.copy()
y_hat_avg['avg_forecast'] = trains['Subscribers'].mean()
rms_simple_spotify = sqrt(mean_squared_error(tests.Subscribers, y_hat_avg.avg_forecast))
print("RMSE Simple Average for spotify: ",rms_simple_spotify)

#Moving average approach
print("Moving Average spotify")
windowsize = 1
y_hat_avg = tests.copy()
y_hat_avg['moving_avg_forecast'] = trains['Subscribers'].rolling(windowsize).mean().iloc[-1]
rms_moving_spotify = sqrt(mean_squared_error(tests.Subscribers, y_hat_avg.moving_avg_forecast))
print("RMSE Moving Average for spotify: ",rms_moving_spotify)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing spotify")
y_hat_avg = tests.copy()
alpha = 1.4
fit2 = SimpleExpSmoothing(np.asarray(trains['Subscribers'])).fit(smoothing_level=alpha,optimized=True)
y_hat_avg['SES'] = fit2.forecast(len(tests))
rms_ses_spotify = sqrt(mean_squared_error(tests.Subscribers, y_hat_avg.SES))
print("RMSE SES for spotify: ",rms_ses_spotify)

# Holt
print("Holt spotify")
sm.tsa.seasonal_decompose(trains.Subscribers,freq = 1).plot()
result = sm.tsa.stattools.adfuller(trains.Subscribers)

y_hat_avg = tests.copy()
alpha = 0.2
fit1 = Holt(np.asarray(trains['Subscribers'])).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(tests))
rms_holt_spotify = sqrt(mean_squared_error(tests.Subscribers, y_hat_avg.Holt_linear))
print("RMSE Holt for spotify: ",rms_holt_spotify)

########################################################

total_rms_simple=rms_simple_apple + rms_simple_spotify
total_rms_moving=rms_moving_apple + rms_moving_spotify
total_rms_ses=rms_ses_apple + rms_ses_spotify
total_rms_holt=rms_holt_apple + rms_holt_spotify
error_total=[]
error_total.extend((total_rms_simple,total_rms_moving,total_rms_ses,total_rms_holt))
if min(error_total) == total_rms_simple:
    print("Min error is from simple average approach")
elif min(error_total) == total_rms_moving:
    print("Min error is from moving average approach")
elif min(error_total) == total_rms_ses:
    print("Min error is from SES")
else:
    print("Min error is from Holt")
    
########################################################

alpha_apple = 0.085
alpha_spotify = 0.2
slope_apple = 3.75
slope_spotify = 0.1
holt_apple = round(estimate_holt(applearray,alpha_apple, slope_apple),4)
print("Holt trend estimation for apple with alpha =", alpha_apple, ", and slope =", slope_apple, ": ", holt_apple)
holt_spotify = round(estimate_holt(spotifyarray,alpha_spotify, slope_spotify),4)
print("Holt trend estimation for spotify with alpha =", alpha_spotify, ", and slope =", slope_spotify, ": ", holt_spotify)