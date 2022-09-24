"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""

#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array import *
import statistics as st
from sklearn.metrics import mean_squared_error as rmse
from sklearn.decomposition import PCA
linalg=np.linalg



#reading csv files
data1=pd.read_csv ("pima-indians-diabetes.csv")
list_col=list(data1.columns)
list_col1=list_col[:-1]

# *************QUESTION 1 ****************** #

print("\n Q1 \n")
#function to find lower and upper range of data
def outlier_rem (array):
    q3,q1 = np.percentile(array, [75,25])
    iqr= q3-q1
    lower_range= q1 - 1.5*iqr
    upper_range= q3 + 1.5*iqr
    return [lower_range,upper_range]

#removing outliers
for list_item in list_col1:
    lower_range, upper_range = outlier_rem(data1[list_item])
    data1.loc[data1[list_item] <lower_range , list_item]=np.nan
    data1.loc[data1[list_item] > upper_range, list_item]= np.nan

#fidning mean for only values under IQR range
for list_item in list_col1:
    lower_range, upper_range = outlier_rem(data1[list_item])
    column=array('f',[])
    for value in data1[list_item]:
        if (value > upper_range):
            continue
        if (value < lower_range):
            continue
        else:
            column.append(value)
    med= st.median(column)
    data1=data1[list_col].fillna(med)

#function to shiow box plot
def showBoxPlot(array, col_name):
    plt.boxplot(array)
    plt.xlabel("Attribute:" + col_name)
    plt.title("Boxplot")
    plt.show()

#QUESTION 1 A)
print("\nA part\n")
#function to find min and max of data
def max_min (array):
    max= np.max(array)
    min= np.min(array)
    return(max,min)

#printing min ,max of data
for i in list_col1:
    max, min = max_min(data1[i])
    print(f" max of {i} ", max )
    print(f" min of {i} ", min)
    print()

#normalisatiob of data
data_normalize=data1.copy()
for column in data_normalize.columns:
    data_normalize[column] =  (((data_normalize[column] - data_normalize[column].min()) /(
        data_normalize[column].max()- data_normalize[column].min()))* ( 12 -5) ) + 5

#printing min max after normalisation
for i in list_col1:
    max, min = max_min(data_normalize[i])
    print(f" after normalisation max of {i} ", max )
    print(f" after normalisation min of {i} ", min )
    print()
#QUESTION 1 B)

print("\n B part\n")
#function to obtain mean, std
def mean_std(array):
    mean=np.mean(array)
    std=np.std(array)
    return (mean,std)

#printning mean,std of data
for i in list_col1:
    mean,std= mean_std(data1[i])
    print(f"mean of {i} is: ",mean)
    print(f"standard deviation of {i} is: ", std)
    print()

#standarization of data
data_standardize=data1.copy()
for column in data_standardize.columns:
    data_standardize[column]=(data_standardize[column] - data_standardize[column].mean())/data_standardize[column].std()

#printing mean and std after standardization of data
for i in list_col1:
    mean,std= mean_std(data_standardize[i])
    print(f"after standize mean of {i} is: ",round(mean,2))
    print(f"after standardize tandard deviation of {i} is: ", round(std,2))
    print()

# ************** QUESTION2************ #

print("\n Q2 \n")
n=1000
mean= [0,0]
cov=[[13,-3],[-3,5]]

#obtaining eigen values and eigen vectors
evalue,evect= np.linalg.eig(cov)
print("eigen values : ", evalue)
print()
print("eigen vectors : ",evect)
origin=[0,0]
evect1= evect[:,0]
evect2= evect[:,1]

#A part
#scatter plot of samples
data12=np.random.multivariate_normal(mean,cov,n)
plt.scatter(data12[:,0], data12[:,1], c="slategrey", marker="+")
plt.title("scatter plot of data sample")
plt.xlabel("column 2")
plt.ylabel("column 1")
plt.show()

# B part
#scatter plot with eigen vectors
plt.scatter(data12[:,0], data12[:,1], c="slategrey", marker="+")
plt.quiver(origin,origin,evect2,evect1,color="darkred",scale=4)
plt.title("scatter plot of data sample with eigen vector")
plt.xlabel("column 2")
plt.ylabel("column 1")
plt.show()

#C part
#to obtain projection of data on eigen vectors
col1=np.matmul(data12,evect1)
col2=np.matmul(data12,evect2)
pro1=[]
pro2=[]

for i in range(col1.size):
    s=col1[i]*evect1
    t=col2[i]*evect2
    pro1.append(s)
    pro2.append(t)

p1=np.array(pro1)
p2=np.array(pro2)

#projection on 1st eigen vector
plt.scatter(data12[:,0], data12[:,1], c="slategrey", marker="+")
plt.scatter(p1[:,0],p1[:,1], color="red")
plt.quiver(origin,origin,evect2,evect1,color="darkred",scale=4)
plt.title("projection on first eigen vector")
plt.xlabel("column 2")
plt.ylabel("column 1")
plt.show()

#projection on 2nd eigen vector
plt.scatter(data12[:,0], data12[:,1], c="slategrey", marker="+")
plt.scatter(p2[:,0],p2[:,1], color="red")
plt.quiver(origin,origin,evect2,evect1,color="darkred",scale=4)
plt.title("projection on 2nd eigen vector")
plt.xlabel("column 2")
plt.ylabel("column 1")
plt.show()

# D part
#recontruction of data and finding error
datanew= np.dot(data12,evect)
datanew1= np.dot(datanew,evect.T)
err=rmse(data12,datanew1)
print("reconstruction error : ",err)

#*************QUESTION 3 **********************#
print("\n Q3 \n")
data_standardize.drop('class',inplace=True,axis=1)
df=data_standardize.copy()

#obtaining eigen values and eigen vectors after making covarince matrix
cov_m=np.cov(df.T)
evalues,evector= np.linalg.eig(cov_m)
lv=list(evalues)
lvec=list(evector)
list_e=["evalue_1","evalue_2","evalue_3","evalue_4","evalue_5","evalue_6","evalue_7","evalue_8"]
lv.sort(reverse=True)

# part A
#reduccing the dimension of the data to 2 by using PCA analysis
pca=PCA(n_components=2)
principle_compponents= pca.fit_transform(df)
principleDF=pd.DataFrame(data=principle_compponents, columns=["PC1","PC2"])
print("variance of data after doing PCA:\n",np.var(principleDF))
print("eigen values: ", lv)

#scattering the new data on plot
plt.scatter(principleDF["PC1"],principleDF["PC2"], c="slategrey", marker="+")
plt.title("scatter plot of reduced dimensional data")
plt.xlabel("principle component 1")
plt.ylabel("principle component 2")
plt.show()

# part B
#ploting the eigen values in descending order
plt.plot(list_e,lv, color="orangered", marker="+")
plt.grid(color='slategrey', linestyle='-.', linewidth=1)
plt.title("bar plot of eigen values")
plt.xlabel("eigen values ")
plt.ylabel(" y axis")
plt.show()

# part C
#PCA analysis on the data for different l values
errors=[]
for i in range(1,9):
    pca=PCA(n_components=i)
    pcom=pca.fit_transform(df)
    pcom_back=pca.inverse_transform(pcom)
    losses=(sum(((df - pcom_back) ** 2).sum(axis=1)) / len(df))**0.5
    errors.append(losses)

#ploting error values agains the no of l values
list2=[1,2,3,4,5,6,7,8]
plt.title("reconstruct error of pca")
plt.plot(list2,errors, color="orangered",marker='+')
plt.xlabel('No of dimenssion (l)')
plt.ylabel('Euclidan distance')
plt.grid(color='slategrey', linestyle='-.', linewidth=1)
plt.show()

#obtaining covarinace matrix for the different l values
def covmax(l):
    pca = PCA(n_components=l)
    pcm = pca.fit_transform(df)
    pcm=pd.DataFrame(pcm)
    cov2= round(pcm.cov(),3)
    print(cov2)
for i in range(1,9):
    print(f"covarinace matrix for l= {i} \n")
    print(covmax(i))

#part D
cova1= np.dot(df.T,df)
print(round(df.cov(),3))






























