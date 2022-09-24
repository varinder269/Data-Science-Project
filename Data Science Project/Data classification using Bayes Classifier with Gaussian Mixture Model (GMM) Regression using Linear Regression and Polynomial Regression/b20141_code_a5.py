"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""

#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as sk
from sklearn.mixture import GaussianMixture as gm
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as asc
from sklearn.preprocessing import PolynomialFeatures as pf

# ***************** Q1 ************** #
print("PART A\n\nQUESTION-1\n")

#reading csv files
data = pd.read_csv("SteelPlateFaults-2class.csv")
data_test = pd.read_csv("SteelPlateFaults-test.csv")
data_train = pd.read_csv("SteelPlateFaults-train.csv")

#removing the sttributes which are highly correlated
data=data.drop(columns=['X_Minimum' , 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis =1)
data_test=data_test.drop(columns=['X_Minimum' , 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis =1)
data_train=data_train.drop(columns=['X_Minimum' , 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis =1)

#dividing data by classes
divided_by = data_train.groupby('Class')
data_train_class0 = divided_by.get_group(0)
data_train_class1 = divided_by.get_group(1)

#droping and obtaining class labels
class0_train = data_train_class0['Class']
class1_train = data_train_class1['Class']
data_train_class0 = data_train_class0.drop(['Class'], axis =1)
data_train_class1 = data_train_class1.drop(['Class'], axis =1)

#obtaining class labels for test data
data_test_class = data_test['Class']
data_test = data_test.drop(['Class'], axis=1)

ac_max= 0
q=0
#GMM model to find the class of test data
for i in [2,4,8,16]:  #loop for different values of the Q clusters
    gmm_c_0 = gm(n_components=i, covariance_type= 'full', reg_covar = 1e-4)  #gmm for different Q values
    gmm_c_0.fit(data_train_class0.values)
    gmm_c_1 = gm(n_components=i, covariance_type= 'full', reg_covar = 1e-4)
    gmm_c_1.fit(data_train_class1.values)

    p = []
    # obtaining the target attributes for test data
    s0= gmm_c_0.score_samples(data_test.values)
    s1= gmm_c_1.score_samples(data_test.values)

    #selecting highest probability class
    for j in range(len(s0)):
        if s0[j]>s1[j]:
            p.append(0)
        if s0[j]<s1[j]:
            p.append(1)

    #making confusion matrix for the data
    matrix = cm(data_test_class.values, p)
    acc = asc(data_test_class.values, p) #getting accuracy score for test data
    print(f"confusion matrix for {i} clusters is : \n",matrix )
    print(f"accuracy score got {i} clusters is: ", round(acc*100,3),"\n")
    if acc > ac_max :
        ac_max=acc
        q=i
print("The maximum accuracy is: ",ac_max*100, " for ",q," clusters")
print("\nQUESTION 2 IS ON REPORT\n\n")
#QUESTION 2 is on report

# **************** PART B ******************#
# **************** Q2 **********************#

print("PART B\n\nQUESTION-1\n")
data_new = pd.read_csv("abalone.csv")

#finding the highest correlation coefficient
ivar = data_new[data_new.columns[1:]].corr()['Rings'][:-1].idxmax()
print("The attribute with highest Pearson correlation coefficient with the target attribute is ",ivar)

data_new_shell = data_new["Shell weight"]
data_new_rings = data_new["Rings"]

#spliting of the data
data_train, data_test, data_label_train, data_label_test = sk.train_test_split(data_new, data_new_rings, test_size=0.3, random_state=42, shuffle=True)
data_test.to_csv('abalone-test.csv')
data_train.to_csv('abalone-train.csv')

data_new_train, data_new_test, data_new_label_train, data_new_label_test = sk.train_test_split(data_new_shell, data_new_rings, test_size=0.3, random_state=42, shuffle=True)

#linear regression
LR = LinearRegression()
LR.fit(data_new_train.values.reshape(-1,1),data_new_label_train.values)  #fitting train data to learn
prediction = LR.predict(data_new_test.values.reshape(-1,1))   #prediction of the test values
prediction1 = LR.predict(data_new_train.values.reshape(-1,1))   #prediction for train values

#Q1 A
plt.style.use('fivethirtyeight')  #plotting the prediction and real values
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_new_train,data_new_label_train,label="Actual test data",marker="*",color='black')   #scattering the data
axs.plot(data_new_train , prediction1,label="Linear regression",color="red",linewidth=4)  #ploting the best fit line
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Line')
plt.legend()
plt.show()

df_train= data_train.copy()
df_test= data_test.copy()

def error(prediction1,df_train):  #defining function for the rms error
    error = 0
    for k in range(0,len(df_train)):
        error += (((prediction1[k]- df_train['Rings'].iloc[k])**2))/len(df_train)
    x=round((error**0.5),3)
    return x

#Q1 B
print("RMSE error for the train data is :",error(prediction1,df_train))

#Q1 C
print("RMSE error for the test data is :",error(prediction,df_test))

#Q1 D
plt.style.use('fivethirtyeight')  #ploting the prediction vs actual values
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_new_label_test,prediction,marker="*",color='black')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title("prediction vs. acutal rings univariate")
plt.show()

# ******************QUESTION 2 ************* #
print('\nQUESTION 2\n')

#Q2 A

#linear regression
LR.fit(data_train.iloc[:, :-1].values,data_label_train.values)  #fitting train data to learn
prediction = LR.predict(data_test.iloc[:, :-1].values)   #prediction of the test values
prediction1 = LR.predict(data_train.iloc[:, :-1].values)   #prediction for train values
print("RMSE error for the train data for multivariate linear regression is :",error(prediction1,df_train))

#Q2 B
print("RMSE error for the test data for multivariate linear regression is :",error(prediction,df_test))

#Q2 C
plt.style.use('fivethirtyeight')  #ploting the real vs. prediction fro multivariate regression
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_label_test,prediction,marker="*",color='black')
plt.title("prediction vs. acutal rings for multivariate")
plt.ylabel('predicted rings')
plt.xlabel("actual rings")
plt.show()

#******************QUESTION 3 ******************#
print('\nQUESTION 3\n')

def nonl_reg (df_train, train, data_use):  #defining function for the polynomail regeression and maing plot of rmse
    data_x = np.array(data_train['Shell weight']).reshape(-1, 1)
    data_x_t = np.array(data_use['Shell weight']).reshape(-1, 1)
    RMSE = []
    for p in [2,3,4,5]:
        poly_features = pf(p)  #seelcting the polynomial degree
        x_poly = poly_features.fit_transform(data_x)
        x_poly_y = poly_features.fit_transform(data_x_t)
        LR= LinearRegression()
        LR.fit(x_poly, data_label_train)  #fitting the data learn
        pred = LR.predict(x_poly_y) #predciton
        rmse = error( pred, df_train)  #error
        RMSE.append(rmse)
        print(f"The rmse for {train} data for polynomial degree ", p, 'is',rmse)

    #plot for rmse vs polynomial degrees
    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    axs.bar([2, 3, 4, 5], RMSE, color='black')
    plt.title("rmse vs. polynomail degree plot")
    plt.ylabel(f'rmse values {train}data')
    plt.xlabel("degree of ploynomial")
    plt.show()
    print()

#Q3 A
nonl_reg(df_train,"train", df_train)

#Q3 B
nonl_reg(df_test,"test", df_test)

#Q3 C
data_x = np.array(data_train['Shell weight']).reshape(-1, 1)
data_x_t = np.array(data_test['Shell weight']).reshape(-1, 1)

#again linaer regression for p=5
poly_features = pf(5)   # p =5 has telowest rmse
x_poly = poly_features.fit_transform(data_x)
x_poly_y = poly_features.fit_transform(data_x_t)
LR= LinearRegression()
LR.fit(x_poly, data_label_train)
pred = LR.predict(x_poly_y)
pred1= LR.predict(x_poly)

plt.style.use('fivethirtyeight')  #plotting the prediction and real values
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_new_train,data_new_label_train,label="Actual test data",marker="*",color='black')
axs.scatter(data_new_train, pred1,label="Linear regression",color="red",linewidth=4) #ploting the best fit line
plt.legend()
plt.show()

#Q3 D
plt.style.use('fivethirtyeight')  #plotting the prediction and real values
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_label_test,pred,label="Actual test data",marker="*",color='black')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()

# ********************QUESTION 4****************#
print('\nQUESTION 4\n')

def nonl_m_reg (df_train, train, data_use):  #defining function for the polynomail regeression and maing plot of rmse
    data_x = data_train.iloc[:, :-1].values
    data_x_t = data_use.iloc[:, :-1].values

    RMSE = []
    for p in [2,3,4,5]:
        poly_features = pf(p)  #seelcting the polynomial degree
        x_poly = poly_features.fit_transform(data_x)
        x_poly_y = poly_features.fit_transform(data_x_t)
        LR= LinearRegression()
        LR.fit(x_poly, data_label_train)  #fitting the data learn
        pred = LR.predict(x_poly_y) #predciton
        rmse = error( pred, df_train)  #error
        RMSE.append(rmse)
        print(f"The rmse for {train} data for multivariate regression for polynomila degree ", p, 'is',rmse)

    #plot for rmse vs polynomial degrees
    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    axs.bar([2, 3, 4, 5], RMSE, color='black')
    plt.title("rmse vs. polynomail for multivariate degree plot")
    plt.ylabel(f'rmse values {train}data')
    plt.xlabel("degree of ploynomial")
    plt.show()
    print()

#Q4 A
nonl_m_reg(df_train,"train", df_train)

#Q4 B
nonl_m_reg(df_test,"test", df_test)

#Q4 C

data_x = data_train.iloc[:, :-1].values
data_x_t = data_test.iloc[:, :-1].values

#again linaer regression for p=2
poly_features = pf(2)   # p =2 has telowest rmse
x_poly = poly_features.fit_transform(data_x)
x_poly_y = poly_features.fit_transform(data_x_t)
LR= LinearRegression()
LR.fit(x_poly, data_label_train)
predm = LR.predict(x_poly_y)

#ploting the actual vs predicted values for the data
plt.style.use('fivethirtyeight')  #plotting the prediction and real values
fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
axs.scatter(data_label_test,predm,label="Actual test data",marker="*",color='black')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('multivariate non-linear regression model')
plt.show()



