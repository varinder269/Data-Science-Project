
"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""

#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#reading the csv file

df= pd.read_csv ("pima-indians-diabetes_aebfd8b3f090ccd0742c1f02ce65e957.csv")

#Question 1

#******calculating Mean, median, mode, minimum, maximum and standard deviation for all the attributes******

#using inbiult functions to find mean,median etc.
print("Mean of different attributes are: ")
print(df.mean())
print()
print("Median of different attributes are: ")
print(df.median())
print()
print("Mode of different attributes are: ")
print(df.mode())
print()
print("Maximum of different attributes are: ")
print(df.max())
print()
print("Minimum of different attributes are: ")
print(df.min())
print()
print("standard deviation of different attributes are: ")
print(df.std())
print()

#Question 2

#reading the different attributes
pregs = df['pregs']
plas = df['plas']
pres = df['pres']
skin = df['skin']
test = df['test']
BMI= df['BMI']
pedi = df['pedi']
Age = df['Age']

#Question 2a

#*************** obtaining scatter plots between 'age' and other attributes *************

#scatter plot between age and pregs

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,pregs, s=100, alpha=0.7, edgecolor='black', linewidth=1)  #creating scatter plot
plt.title('Plot between Age and No. of times Pregnent') # giving title
plt.xlabel('Age in yrs') #labeling of x axis
plt.ylabel('no. of times pregnent') #labeling of y axis
plt.tight_layout()
plt.show()

#scatter plot between age and plasma conc.

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,plas, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and Plasma glucose concentration ')
plt.xlabel('Age in yrs')
plt.ylabel(' Plasma glucose concentration 2 hrs in an oral glucose tolerance test')
plt.tight_layout()
plt.show()

#scatter plot between age and blood pressure

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,pres, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and blood pressure ')
plt.xlabel('Age in yrs')
plt.ylabel('Diastolic blood pressure (mm Hg) ')
plt.tight_layout()
plt.show()

#scatter plot between age and skin thickness

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,skin , s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and skin fold thickness')
plt.xlabel('Age in yrs')
plt.ylabel('Triceps skin fold thickness (mm)')
plt.tight_layout()
plt.show()

#scatter plot between age and test: 2-Hour serum insulin

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,test, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and 2-Hour serum insulin')
plt.xlabel('Age in yrs')
plt.ylabel('test: 2-Hour serum insulin (mu U/mL)')
plt.tight_layout()
plt.show()

#scatter plot between age and BMI

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,BMI, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and BMI')
plt.xlabel('Age in yrs')
plt.ylabel(' Body mass index (weight in kg/(height in m)^2) ')
plt.tight_layout()
plt.show()

#scatter plot between age and pedigree

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(Age,pedi, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and pedigree')
plt.xlabel('Age in yrs')
plt.ylabel('Diabetes pedigree function ')
plt.tight_layout()
plt.show()


#Question 2b

#***************  obtaining scatter plots between 'age' and other attributes***********

#scatter plot between BMI and pregs

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,pregs, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and No. of times Pregnent')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel('no. of times pregnent')
plt.tight_layout()
plt.show()

#scatter plot between BMI and plasma conc.

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,plas, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and Plasma glucose concentration ')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel(' Plasma glucose concentration 2 hrs in an oral glucose tolerance test')
plt.tight_layout()
plt.show()

#scatter plot between BMI and blood pressure

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,pres, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and blood pressure ')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel('Diastolic blood pressure (mm Hg) ')
plt.tight_layout()
plt.show()

#scatter plot between BMI and skin thickness

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,skin , s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and skin fold thickness')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel('Triceps skin fold thickness (mm)')
plt.tight_layout()
plt.show()

#scatter plot between BMI and test: 2-Hour serum insulin

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,test, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and 2-Hour serum insulin')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel('test: 2-Hour serum insulin (mu U/mL)')
plt.tight_layout()
plt.show()

#scatter plot between BMI and pedigree

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,pedi, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between Age and pedigree')
plt.xlabel('Body mass index (weight in kg/(height in m)^2) ')
plt.ylabel('Diabetes pedigree function ')
plt.tight_layout()
plt.show()

#scatter plot between BMI and age

plt.style.use('seaborn') #to get seaborn scatter plot
plt.scatter(BMI,Age, s=100, alpha=0.7, edgecolor='black', linewidth=1)
plt.title('Plot between BMI and Age')
plt.xlabel('Age in yrs')
plt.ylabel('Body mass index (weight in kg/(height in m)^2) ')
plt.tight_layout()
plt.show()

#Question 3

#Question 3a
#obtaining correlation coefficient between age and other attributes

#using inbuilt correlation function
print("Correlation between Age and pregs is ",df['Age'].corr(df['pregs']))
print("Correlation between Age and plas is ",df['Age'].corr(df['plas']))
print("Correlation between Age and pres is ",df['Age'].corr(df['pres']))
print("Correlation between Age and skin is ",df['Age'].corr(df['skin']))
print("Correlation between Age and test is ",df['Age'].corr(df['test']))
print("Correlation between Age and BMI is ",df['Age'].corr(df['BMI']))
print("Correlation between Age and pedi is ",df['Age'].corr(df['pedi']))
print("Correlation between Age and Age is ",df['Age'].corr(df['Age']))
print()

#Question 3b
#obtaining correlation coefficient between age and other attributes

print("Correlation between BMI and pregnent is ",df['BMI'].corr(df['pregs']))
print("Correlation between BMI and plasma is ",df['BMI'].corr(df['plas']))
print("Correlation between BMI and pressure is ",df['BMI'].corr(df['pres']))
print("Correlation between BMI and skin is ",df['BMI'].corr(df['skin']))
print("Correlation between BMI and test is ",df['BMI'].corr(df['test']))
print("Correlation between BMI and pedigree is ",df['BMI'].corr(df['pedi']))
print("Correlation between BMI and Age is ",df['BMI'].corr(df['Age']))
print("Correlation between BMI and BMI is ",df['BMI'].corr(df['BMI']))

#Question 4
#ploting histogram for pregs and skin

#for "pregs"
plt.hist(df['pregs'],bins=[0,2,4,6,8,10,12,14,16,18,20],edgecolor="black",color="orange") #creating a histogram
plt.xlabel("no. of times pregnent") #labeling of x axis
plt.ylabel("No. of diabetics patients") #labeling of y axis
plt.title("histogram for the 'Pregs'") # giving title
plt.show()

#for "skin'
plt.hist(df['skin'],bins=[0,10,20,30,40,50,60,70,80,90,100],edgecolor="black",color="orange")
plt.xlabel("thickness of skin")
plt.ylabel("no. of diabetics patients")
plt.title('histogram for the "skin"')
plt.show()

#Q5

#ploting histogram for 2 different classes for 'pregs'

group=df.groupby('class')
for i,j in group:   # i is the class and j is data corrosponds to class
  plt.hist(j['pregs'],bins=[0,2,4,6,8,10,12,14,16,18,20],edgecolor='black', color="orange") #creating histogram
  plt.title(f'histogram for class :{i}') #title of hist
  plt.ylabel("no. of diabetics patients")
  plt.xlabel("no. of times pregnent")
  plt.show()

#Q6

# boxplot for all atrributes

plt.boxplot(df['pregs'])  #creating box plot
plt.title("boxplot for attribute 'pregs'") # title of boxplot
plt.ylabel("No. of times pregnent") # y label
plt.show()

plt.boxplot(df['plas'])
plt.title("boxplot for attribute 'plas'")
plt.ylabel("plasma Glucose conc.")
plt.show()

plt.boxplot(df['pres'])
plt.title("boxplot for attribute 'pres'")
plt.ylabel("blood pressure (mm Hg)")
plt.show()

plt.boxplot(df['skin'])
plt.title("boxplot for attribute 'skin'")
plt.ylabel("Triceps skin fold thickness (mm)")
plt.show()

plt.boxplot(df['test'])
plt.title("boxplot for attribute 'test'")
plt.ylabel(" 2-Hour serum insulin (mu U/mL)")
plt.show()

plt.boxplot(df['BMI'])
plt.title("boxplot for attribute 'BMI'")
plt.ylabel("Body mass index (weight in kg/(height in m)^2) ")
plt.show()

plt.boxplot(df['pedi'])
plt.title("boxplot for attribute 'pedi'")
plt.ylabel("Diabetes pedigree function")
plt.show()

plt.boxplot(df['Age'])
plt.title("boxplot for attribute 'Age'")
plt.ylabel("Age in years")
plt.show()



