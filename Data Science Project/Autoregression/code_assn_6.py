"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""

#importing modules
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
import math
from statsmodels.tsa.ar_model import AutoReg

#**************QUESTION 1 *****************#

print("\nQUESTION 1")

print("\nPART A\nScatter plot")
# reading the csv file
data = pd.read_csv('daily_covid_cases.csv')
new_data = data['new_cases']

#creating a plot between days and covid cases
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)

xticks = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21'] #fixing x ticks
plt.xticks([i for i in range(int(612/11),612,int(612/11)) ], xticks, rotation = 45)
plt.plot(data['Date'] , data['new_cases'],color="red",linewidth=3)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.title('Cases vs Months')
plt.show()

print("\nPART B")
# generating time series with 1 day lag
lagged = data['new_cases'].shift(1)
corr = pearsonr(lagged[1:], new_data[1:])
print("coefficient for the generated one-day lag time sequence and the given time sequence is ",round(corr[0], 3))

print("\nPART C\nScattered plot ")
#scattering the lag series vs. real time series
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(new_data[1:], lagged[1:],marker="*",color='black')   #scattering the data
plt.xlabel('given time sequence')
plt.ylabel('one day lagged time sequence')
plt.title('lag sequence vs. real time sequence')
plt.show()

print("\nPART D")
#Finding correlation for different lag values
l = [1, 2, 3, 4, 5, 6]
correlation = []
for i in l:
    lagged = data['new_cases'].shift(i)
    corr = pearsonr(lagged[i:], new_data[i:])
    correlation.append(corr[0])
    print(f"The correlation coefficient for {i}-day lag series and the real time sequence is ",round(corr[0],3))

#ploting correlation for different p values
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
plt.plot(l, correlation,color="red",linewidth=3)
plt.xlabel('Lag values')
plt.ylabel('Correlation coefficients')
plt.title('Correlation coefficients vs Lag values')
plt.show()

print("\nPART E\nCorrelogram")
#making a correlogram for diffrent lag values
plt.style.use('fivethirtyeight')
sm.graphics.tsa.plot_acf(new_data,lags=l,color="maroon",linewidth=3,vlines_kwargs={"colors": "red"})
plt.xlabel('Lag Values')
plt.ylabel('Correlation coefficients')
plt.show()

#**************QUESTION 2 *****************#

print("\nQUESTION 2")
#splitting the data
df = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 #35% for testing
X = df.values
tst_sz = math.ceil(len(X)*test_size)
train_df, test_df = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

print("\nPART A")
# training the model
p = 5
model = AutoReg(train_df, lags=p)
model_fit = model.fit()  # train the model
coef = model_fit.params # Get the coefficients of AR model
print('The coefficients obtained from the AR model are:\n', coef) # printing the coefficients

print("\nPART B")
def auto_reg(p,train_df,test_df,coef):
    histor = train_df[len(train_df)-p:]
    history = [histor[i] for i in range(len(histor))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test_df)):
        length = len(history)
        lag = [history[i] for i in range(length-p,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(p):
            yhat += coef[d+1] * lag[p-d-1] # Add other values
        obs = test_df[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    return predictions,history

predictions,history = auto_reg(p,train_df,test_df,coef)

print("\n(i)\nPLOT")
# scatter plot between actual and predicted values
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(predictions, test_df,marker="*",color='black')   #scattering the data
plt.xlabel('Prediction Values')
plt.ylabel('Actual Values')
plt.title('Actual vs Predicted values')
plt.show()

print("\n(ii)\nPLOT")
#creating a line plot between days actual and predicted values
df = pd.read_csv('daily_covid_cases.csv')
l = len(df["new_cases"])
train_size = int(l*0.65)
train = list(df.iloc[:train_size, 1])
test = df.iloc[train_size:, 1]

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.plot([i for i in range(train_size,l)], predictions,label="predictions", color="red",linewidth=5)   #ploting predictions curve
axs.plot([i for i in range(train_size, l)], test, label="actual",color="blue",linewidth=2) #ploting real values curve
plt.xlabel("NO of days after 30-Jan")
plt.ylabel("No of co-vid cases")
plt.title("Test data")
plt.show()

print("\n(iii)")
n=len(test_df)
s=0  # for rmse
for i in range(n):
    s=s+(predictions[i]-test_df[i])**2
avg=sum(test_df)/len(test_df)
rmse=(math.sqrt(s/len(test_df))/avg)*100
print("Rmse = ",rmse)

s=0 #For MAPE
for i in range(n):
    s=s+ abs(predictions[i]-test_df[i])/test_df[i]
mape=(s/n)*100
print("MAP = ",mape)

print("\nQUESTION 3")

#defining function for autoregression and rmse/mape errors
def auto_regression(i):
      model = AutoReg(train_df, lags=i)
      # fit/train the model
      model_fit = model.fit()
      coef = model_fit.params
      history = train_df[len(train_df)-i:]
      history = [history[j] for j in range(len(history))]
      predictions = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
      for t in range(len(test_df)):
        length = len(history)
        Lag = [history[i] for i in range(length-i,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(i):
            yhat += coef[d+1] * Lag[i-d-1] # Add other values
        obs = test_df[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.

      n = len(test_df)
      s = 0 # RMSE Calculation
      for i in range(n):
          s = s + (predictions[i] - test_df[i]) **2
      avg = sum(test_df) / len(test_df)
      rmse = (math.sqrt(s / len(test_df)) / avg) *100

      s = 0  # MAPE Calculation
      for i in range(n):
          s = s + abs(predictions[i] - test_df[i]) / test_df[i]
      mape = (s / n) * 100
      return rmse[0], mape[0]

rmse=[0]*5
mape=[0]*5
p = []

#obtaining results pf rmse and mape for idfferent values of p
rmse[0] , mape[0] = auto_regression(1)
p.append(1)
for i in range (1,4):
        rmse[i] , mape[i] = auto_regression(5*i)
        p.append(5*i)
rmse[4], mape[4] = auto_regression(25)
p.append(25)

#RMSE(%) and MAPE between predicted and original data values fo different lags
data = {'p value':p,'RMSE(%)':rmse, 'MAPE' :mape}
print(pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
axs.bar([1,2, 3, 4, 5], rmse, color='black')
plt.xlabel('Time Lag')
plt.ylabel('RMSE(%)')
plt.title('RMSE(%) vs. time lag')
plt.xticks([1,2,3,4,5],p)
plt.show()

# plotting MAPE vs. time lag
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
axs.bar([1,2, 3, 4, 5], mape, color='black')
plt.xlabel('Time Lag')
plt.ylabel('MAPE')
plt.title('MAPE vs. time lag')
plt.xticks([1,2,3,4,5],p)
plt.show()

print("\nQUESTION 4")
#finding the optimum value of p
p=1
flag=1
while(flag==1):
    new_train=train_df[p:]
    l=len(new_train)
    lag_new_train=train_df[:l]
    nt =[]
    lnt =[]
    for i in range (len(new_train)):
        nt.append(new_train[i][0])
        lnt. append(lag_new_train[i][0])
    corr = pearsonr(lnt,nt)
    if(2/math.sqrt(l)>abs(corr[0])): #condition for optimum p
        flag=0
    else:
        p=p+1
print('The heuristic value for the optimal value of p is',p-1)
x = p-1
model = AutoReg(train_df, lags=x)
model_fit = model.fit()  # train the model
coefi = model_fit.params # Get the coefficients of AR model
predictions,history = auto_reg(x,train_df,test_df,coefi)

n=len(test_df)
s=0 #for rmse
for i in range(n):
    s=s+(predictions[i]-test_df[i])**2
avg=sum(test_df)/len(test_df)
rmse=(math.sqrt(s/len(test_df))/avg)*100
print("rmse for optimmum lags is: ",rmse)
s=0 # For MAPE
for i in range(n):
    s=s+ abs(predictions[i]-test_df[i])/test_df[i]
mape=(s/n)*100
print("mape for optimum lags is: ",mape)







