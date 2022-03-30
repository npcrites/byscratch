# Import Your code
from byscratch.linear_algebra import Vector, Matrix
from byscratch.linear_algebra import make_matrix
from byscratch.linear_algebra import sum_of_squares
from byscratch.linear_algebra import dot
from byscratch.linear_algebra import subtract
from byscratch.linear_algebra import magnitude
from byscratch.linear_algebra import scalar_multiply
from byscratch.linear_algebra import vector_mean
from byscratch.linear_algebra import distance
from byscratch.linear_algebra import add

from byscratch.statistics import correlation
from byscratch.statistics import standard_deviation
from byscratch.statistics import median
from byscratch.statistics import mean
from byscratch.statistics import de_mean
from byscratch.statistics import standard_deviation



from byscratch.gradient_descent import gradient_step

from byscratch.probability import inverse_normal_cdf

from byscratch.working_with_data import rescale


# python library imports
import random, datetime, re, csv, math, enum
from collections import defaultdict, Counter, OrderedDict
from typing import Tuple, List, NamedTuple, Optional, Callable
from typing import TypeVar, List, Iterator

# external code
from dateutil.parser import parse
import tqdm

# pyplot configs
import seaborn as sns
from matplotlib import pyplot as plt

# font
plt.rcParams.update({'font.size': 8})

# reset the default figsize value
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

# 144 is good for a high-resolution display. Try 100 if it's too big
plt.rcParams["figure.dpi"] = (80)

import pandas as pd

df = pd.read_csv('/Users/nickcrites/Desktop/byscratch/Data/bike_sharing_daily.csv')

df.shape

df

df.columns

#remove dteday and replace with day
for i,date in enumerate(df['dteday']):
    df['dteday'][i] = i
df


df

from collections import Counter

data_dict = csv.DictReader(open('/Users/nickcrites/Desktop/byscratch/Data/bike_sharing_daily.csv'))


temp = []
atemp = []
hum = []
windspeed = []
casual_1 = []
reg = []
count = []
workingday = []


for row in data_dict:
    tem = float(row["temp"])
    temp.append(tem)

    avgtemp = float(row["atemp"])
    atemp.append(avgtemp)

    humidity = float(row["hum"])
    hum.append(humidity)

    wndspd = float(row["windspeed"])
    windspeed.append(wndspd)

    cas = float(row["casual"])
    casual_1.append(cas)
    
    cnt = float(row['cnt'])
    count.append(cnt)
    
    wd = float(row['workingday'])
    workingday.append(wd)
    
    r = float(row['registered'])
    reg.append(r)


print(f"The mean temperature is {mean(temp)}, median is {median(temp)}")

print(f"The mean humidity is {mean(hum)}, median is {median(hum)}")

print(f"The mean bike riders per day is {mean(count)}, median is {median(count)}")

#plot users against weather data
sns.lineplot(
     y=temp,
     x=count)
sns.lineplot(
     y=hum,
     x=count)
sns.lineplot(
     y=windspeed,
     x=count)
sns.lineplot(
     y=atemp,
     x=count)



plt.title("Riders vs Temp")
plt.show()

#plot users against weather data
sns.lineplot(
     y=temp,
     x=reg)
sns.lineplot(
     y=hum,
     x=reg)
sns.lineplot(
     y=windspeed,
     x=reg)
sns.lineplot(
     y=atemp,
     x=reg)



plt.title("Riders vs Temp")
plt.show()

#plot users against weather data
sns.lineplot(
     y=temp,
     x=casual_1)
sns.lineplot(
     y=hum,
     x=casual_1)
sns.lineplot(
     y=windspeed,
     x=casual_1)
sns.lineplot(
     y=atemp,
     x=casual_1)



plt.title("Riders vs Temp")
plt.show()

df

df = df.drop(['registered','instant','dteday'],axis = 1)
df = df.drop(['casual'], axis = 1)

df

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# DecisionTree Time!

y=df['cnt'].values
df = df.drop(['cnt'], axis =1)
X=df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

decisionTree = DecisionTreeRegressor()
decisionTree.fit(X_train, y_train)

decisionTree.score(X_train,y_train), decisionTree.score(X_test,y_test)


decisionTree.predict(X_test)

y_test

from sklearn import tree

tree.plot_tree(decisionTree)

decisionTree.feature_importances_

decisionTree.score(X_train,y_train), decisionTree.score(X_test,y_test)


feat_importance= {}
for i, value in enumerate(decisionTree.feature_importances_):
    feat_importance[df.columns[i]] = value
feat_importance

#RandomForest

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

clf.score(X_train,y_train), clf.score(X_test,y_test)

clf.feature_importances_

RF_feat_importance= {}
for i, value in enumerate(clf.feature_importances_):
    RF_feat_importance[df.columns[i]] = value
RF_feat_importance

