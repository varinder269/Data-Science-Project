"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""
#importing modules
import pandas as pd
import numpy as np
from array import *
import sklearn.model_selection as sk
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as asc

#reading csv files
df= pd.read_csv('SteelPlateFaults-2class.csv', delimiter=",")
list_col = list(df.columns)

#*****************QUESTION 1 ****************#
print("\n QUESTION 1 \n")

#dividing data by classes
datanew= df.groupby('Class')
datanew0= datanew.get_group(0)
datanew1= datanew.get_group(1)

#spliting of the data
datanew0_train, datanew0_test, datanew0_label_train, datanew0_label_test = sk.train_test_split(datanew0, datanew0['Class'], test_size=0.3, random_state=42, shuffle=True)
datanew1_train, datanew1_test, datanew1_label_train, datanew1_label_test = sk.train_test_split(datanew1, datanew1['Class'], test_size=0.3, random_state=42, shuffle=True)

#making dataframes for test and training data and then making csv files
data_train= pd.concat([pd.DataFrame(datanew0_train),pd.DataFrame(datanew1_train)]) #making dataframes from arrays and then concating them
data_train.to_csv('SteelPlateFaults-train.csv',index=False)  #making csv files doe rhe data
data_test=pd.concat([pd.DataFrame(datanew0_test),pd.DataFrame(datanew1_test)])
data_test.to_csv('SteelPlateFaults-test.csv',index=False)

#making data frames for labels of training and test data
data_train_label = pd.concat([datanew0_label_train ,datanew1_label_train])
data_test_label = pd.concat([datanew0_label_test,datanew1_label_test])

#reading our new csv files
data_train= pd.read_csv('SteelPlateFaults-train.csv')
data_test= pd.read_csv('SteelPlateFaults-test.csv')
data_train.drop('Class',inplace=True,axis=1)
data_test.drop('Class',inplace=True,axis=1)

# K-nearest neighnor (KNN)
arr=array('f',[])
h_acc_K = 1
h_acc_value = 0
for i in range(1,6,2):
    classifier= KNC(n_neighbors=i)  #clasifier for diffrent k values
    classifier.fit(data_train, data_train_label) #fitting the data for classifier
    prediction = classifier.predict(data_test) #prediction for target attribute for test data
    con_max= cm(data_test_label,prediction) #obtaining confusion matrix
    accuracy_sc = asc(data_test_label,prediction)  #finding the accuracy for the target attributes
    print(f"for k= {i} accuracy score is: ",str(accuracy_sc*100)[0:6:1]," %")
    print(f"the confusion matrix for k={i} is: \n ",con_max,"\n")
    arr.append(accuracy_sc)
    if (accuracy_sc > h_acc_value):
        h_acc_K = i
        h_acc_value = accuracy_sc
print("Highest Accuracy Value is :", str(h_acc_value*100)[0:6:1], "%", " for K :", h_acc_K,"\n")

#*****************QUESTION 2*********************#
print("\n QUESTION2 \n")

#reading csv files
df_train=pd.read_csv('SteelPlateFaults-train.csv')
df_test=pd.read_csv('SteelPlateFaults-test.csv')

data_normalized_train=df_train.copy()
data_normalized_test=df_test.copy()

#normalising the data n the range of 0 to 1
list_a = list(data_normalized_train.columns)
list_a =list_col[:-1] #list of columns except class
for column in list_a:
    data_normalized_train[column] = (data_normalized_train[column] - data_normalized_train[column].min()) /(
        data_normalized_train[column].max()- data_normalized_train[column].min())

#making a csv file for the new train data
data_normalized_train.to_csv('SteelPlateFaults-train-Normalised.csv',index= False)

#normalising the test data using min max from train dsta
for column in list_a:
    data_normalized_test[column] = (data_normalized_test[column] - df_train[column].min()) /(
        df_train[column].max()- df_train[column].min())

#making a csv file for the new test data
data_normalized_test.to_csv('SteelPlateFaults-test-Normalised.csv',index=False)

#reading new csv files after normalisation
data_normalized_train= pd.read_csv('SteelPlateFaults-train-Normalised.csv')
data_normalized_test= pd.read_csv('SteelPlateFaults-test-Normalised.csv')

#storing class labels
dttr= data_normalized_train['Class']
dtte= data_normalized_test['Class']

#dropping class attribute
data_normalized_test.drop('Class',inplace=True,axis=1)
data_normalized_train.drop('Class',inplace=True,axis=1)

# K-nearest neighnor (KNN)
arr_n=array('f',[])
h_acc_K_n = 1
h_acc_value_n=0
for i in range(1,6,2):
    classifier_n= KNC(n_neighbors=i)  #clasifier for diffrent k values
    classifier_n.fit(data_normalized_train, dttr) #fitting the data for classifier
    prediction = classifier_n.predict(data_normalized_test) #prediction for target attribute for test data
    con_max_n= cm(dtte,prediction) #obtaining confusion matrix
    accuracy_sc_n = asc(dtte,prediction)  #finding the accuracy for the target attributes
    print(f" After normalisationnfor k= {i} accuracy score is: ",str(accuracy_sc_n*100)[0:6:1] , " %")
    print(f"confusion matrix for k={i} is : \n",con_max_n,"\n")
    arr_n.append(accuracy_sc_n)
    if (accuracy_sc_n > h_acc_value_n):
        h_acc_K_n = i
        h_acc_value_n = accuracy_sc_n

print("Highest Accuracy Value after normalisation is :", str(h_acc_value_n*100)[0:6:1], "%", " for K :", h_acc_K_n,"\n")

#********************QUESTION3****************#
print("\n QUESTION3\n")
pd.set_option('mode.chained_assignment', None)
#reading new csv files again
data_train_b= pd.read_csv('SteelPlateFaults-train.csv')
data_test_b= pd.read_csv('SteelPlateFaults-test.csv')

#printing mean for each class
data__01_mean=data_train_b.copy()
d_t_mean= data__01_mean.groupby('Class')
data_mean_0= d_t_mean.get_group(0)
data_mean_1= d_t_mean.get_group(1)
# print("the mean for class 0 is:\n", round(data_mean_0.mean(),3),"\n")
# print("the mean for class 1 is:\n", round(data_mean_1.mean(),3),"\n")

#printing covarinance for each class
# print("the cov for class 0 is:\n", round(data_mean_0.cov(),3),"\n")
# print("the cov for class 1 is:\n", round(data_mean_1.cov(),3),"\n")

data_label_test_b=data_test_b['Class']
data_test_b.drop('Class',inplace=True, axis=1)  #dropping the class column

#removing the highly correlated attriuutes
data_train_b.drop(columns=['X_Minimum', 'Y_Minimum','TypeOfSteel_A300', 'TypeOfSteel_A400'], inplace=True)
data_test_b.drop(columns=['X_Minimum', 'Y_Minimum','TypeOfSteel_A300', 'TypeOfSteel_A400'], inplace=True)

#dividing data by classes
df_t= data_train_b.groupby('Class')
datatrain0= df_t.get_group(0)
datatrain1= df_t.get_group(1)

#removing class attributes
datatrain0.drop('Class',inplace=True, axis=1)
datatrain1.drop('Class',inplace=True, axis=1)

#making cov matrix
cov0 = np.cov(datatrain0.T)
cov1 = np.cov(datatrain1.T)
mean0 = np.mean(datatrain0)
mean1 = np.mean(datatrain1)

#getting prior probabilty
prior0 = len(datatrain0)/len(data_train_b)
prior1 = len(datatrain1)/len(data_train_b)

def likelihood(data_vec, mean_v, covariance):
    Matrix = np.dot((data_vec-mean_v).T,np.linalg.inv(covariance))
    ex = np.exp(-0.5*np.dot(Matrix,(data_vec-mean_v)))
    return (ex/((2*np.pi)**11.5 * (abs(np.linalg.det(covariance)))**0.5))

#we dont need to calculate the evidebce (total probabilty ) as it wont effectthe finalanswer becuase it has same weightage in both classes AND its hard to calculate

# obtaing the class which has highest probabilty
LE = []
for i,row_data in data_test_b.iterrows():
    PRO0 = likelihood(row_data, mean0, cov0)*prior0
    PRO1 = likelihood(row_data, mean1, cov1)*prior1
    if(PRO0 > PRO1):
        LE.append(0)
    else:
        LE.append(1)
accuracy_bayes= asc(data_label_test_b,LE) #obraing accuracy for bayes classifier

#priting confusion matrix
print("Confusion matrix for bayes classifier is : \n", cm(data_label_test_b,LE),"\n")
print("Accuracy of bayes classifier is: ", str(accuracy_bayes*100)[0:6:1],"\n")

#****************QUESTION4******************#
print("\n QUESTION4\n")
print("Accuracy for KNN :", str(h_acc_value*100)[0:6:1] + "%")
print("Accuracy for KNN after normaisation :", str(h_acc_value_n*100)[0:6:1]+" %")
print("Accuracy for Bayes Classifier :", str(accuracy_bayes*100)[0:6:1]+" %")









