"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""
#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading csv files
data1=pd.read_csv ("landslide_data3_miss.csv")
data2=pd.read_csv ("landslide_data3_original.csv")


#**************QUESTION 1***********
print("Q1\n")
print(data1.isnull().sum()) #printing no. of missing values
print(data1.isnull().sum().sum())

list_col=list(data1.columns) #storing column names
new_data1=data1.notnull()  #new data frame with boolean expression for data
list_miss=[]

for j in range(len(list_col)):  #loop to get columns
    count=0
    for i in new_data1[f'{list_col[j]}']: #loop to get number of missing values
        if i==False:
            count+=1
    list_miss.append(count)
    j+=1

#ploting barplot
fig=plt.figure(figsize=(10,7))
plt.bar(list_col, list_miss, color="maroon", width= 0.6)
plt.xlabel("atrributes")
plt.ylabel("no. of missing values")
plt.title("No. of missing values per attribute")
plt.show()


#*******QUESTION 2************
print("Q2\n")
# A part
count=0
a=data1['stationid'].notnull() #making boolean from attribute stationid
for i in range(945): #checking for every row
    if a[i]==False: #if values are missing
        data1.drop(i,axis=0,inplace=True) #dropping missing id tuples
        count+=1
print("No. of values dropped: ",count)

#B part
notna = data1.notnull()
clean = data1.dropna(thresh=7)
dele = data1.shape[0] - clean.shape[0]
print("no, of rows dropped: ",dele)
data1=clean
datax=clean
dd=clean.notnull()

#***********QUESTION 3*********

print("Q3\n")
print("the no. of missing values: ")
print(clean.isnull().sum())
print("total no. of missing values",clean.isnull().sum().sum())


#************QUESTION 4*************
print("Q4\n")

#4a (i)
print("4a\n")
#filling missing values with mean of attributes
data1=data1.fillna(data1.mean())

#finding statistics for new data
print("mean,meadian,mode and standard deviation of new file")

print("mean:\n",data1.mean())
print("median:\n",data1.median())
print("mode:\n",data1.mode())
print("standard deviation:\n",data1.std(),"\n")


#obtaining for old data or original file
print("mean,meadian,mode and standard deviation of original file")
print("mean:\n",data2.mean())
print("median:\n",data2.median())
print("mode:\n",data2.mode())
print("standard deviation:\n",data2.std(),"\n")

#4a(ii)
list_col1=list_col[2:] #slicing of columns list to remove dat and station id
RMSE_nos=[]
for col_name in (list_col1): #for each column loop
    sum=0
    count=0
    for i in dd.index.values.tolist(): #for every row in that column loop
        if dd.at[i, col_name] == False: #for missing values
            sum+=((data1.at[i, col_name] -
                             data2.at[i, col_name]) ** 2) #sum of squares
            count+=1 #count
        else:
            continue
    RMSE_nos.append(np.sqrt((sum/count)))
    print(f"RMSE no. for {col_name}",np.sqrt((sum/count)),'\n') #printing sqrt


#ploting barplot
fig=plt.figure(figsize=(10,7))
plt.bar(list_col1,RMSE_nos, color="maroon", width= 0.6) #plot of RMSE AND attributes
plt.xlabel("atrributes")
plt.ylabel("RMSE No. values for mean")
plt.title("RMSE NO. per attribute")
plt.show()

#4b(i)

print("4b\n")
#filling missing values with interpolation method
datax=datax.interpolate(metthod='linear', limit_direction='forward')

#finding statistics for new data
print("mean,meadian,mode and standard deviation of new file")

print("mean:\n",datax.mean())
print("median:\n",datax.median())
print("mode:\n",datax.mode())
print("standard deviation:\n",datax.std(),"\n")

#obtaining for old data or original file
print("mean,meadian,mode and standard deviation of original file")
print("mean:\n",data2.mean())
print("median:\n",data2.median())
print("mode:\n",data2.mode())
print("standard deviation:\n",data2.std(),"\n")

#4b(ii)

print("\n4b\n")

RMSE_nos2=[]
for col_name in (list_col1): #for each column loop
    sum=0
    count=0
    for i in dd.index.values.tolist(): #for every row in that column loop
        if dd.at[i, col_name] == False: #for missing values
            sum+=((datax.at[i, col_name] -
                             data2.at[i, col_name]) ** 2) #sum of squares
            count+=1 #count
        else:
            continue
    RMSE_nos2.append(np.sqrt((sum/count)))
    print(f"RMSE no. for {col_name}",np.sqrt((sum/count)),'\n') #printing sqrt


#ploting barplot
fig=plt.figure(figsize=(10,7))
plt.bar(list_col1,RMSE_nos2, color="maroon", width= 0.6) #plot of RMSE AND attributes
plt.xlabel("atrributes")
plt.ylabel("RMSE No. values for linear interploate")
plt.title("RMSE NO. per attribute")
plt.show()

#**********QUESTION5*************


print("\nquestion 5\n")


#5(i)
plt.boxplot(datax['temperature'])  #creating box plot
plt.title("boxplot for attribute 'temeprature'") # title of boxplot
plt.ylabel("values of temp") # y label
plt.show()

plt.boxplot(datax['rain'])
plt.title("boxplot for attribute 'rain'")
plt.ylabel("values of rain")
plt.show()

#counting outliers
count=0
for i in range(891):
    if datax.iloc[i]['temperature']<9:
        count+=1
print("NO. of outliers in temperature Attribute: ",count)
print()
count=0
for i in range(891):
    if datax.iloc[i]['rain']>2500:
        count+=1
print("NO. of outliers in rain Attribute: ",count)
print()
#5(ii)

#printing row nos. for outliers
print("list of row no.s for outliers in temperature: ",np.where(datax['temperature']<9))
print()
print("list of row nos. for outliers in rain: ",np.where(datax['rain']>2500))
print()

#printing outliers
lista=[]
for i in range(891):
    if datax.iloc[i]['temperature']<9:
        lista.append(datax.iloc[i]['temperature'])
print("list of outliers in temperature: ",lista)
print()
lista=[]
for i in range(891):
    if datax.iloc[i]['rain']>2500:
        lista.append(datax.iloc[i]['rain'])
print("list of outliers in rain : ",lista)
print()


#removing outliers
datax.loc[datax.temperature<9, 'temperature']=np.nan
datax.loc[datax.rain>2500, 'rain']=np.nan
datax=datax.fillna(datax.median()) #filling with median of data

#makig newe box plots after replacing outliers
plt.boxplot(datax['temperature'])  #creating box plot
plt.title("boxplot for attribute 'temeprature' after replacing outliers ") # title of boxplot
plt.ylabel("values of temp") # y label
plt.show()

plt.boxplot(datax['rain'])
plt.title("boxplot for attribute 'rain' after replacing outliers")
plt.ylabel("values of rain")
plt.show()


#couting outliers after part a:
count=0
for i in range(891):
    if datax.iloc[i]['temperature']< 9:
        count+=1
print("NO. of outliers in temperature Attribute after replacing : ",count)
count=0
for i in range(891):
    if datax.iloc[i]['rain']>157.7:
        count+=1
print("NO. of outliers in rain Attribute after replacing : ",count)


print()
print("list of row nos. for outliers in rain: ",np.where(datax['rain']>157.7))
print()



