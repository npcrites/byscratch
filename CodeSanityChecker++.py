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

df = pd.read_csv('byscratch/Data/adult_original.csv')

df_less = df.loc[df['income'] == '<=50K'].sample(n=11200)
df_more = df.loc[df['income'] == '>50K'].sample(n=11200)

df_less.shape, df_more.shape

df_stratifiedincome = pd.concat([df_less,df_more])
df_stratifiedincome.to_csv('byscratch/data/adult_stratified_income.csv')

df_stratifiedincome.shape

# show the first two rows: header and data
!head -1 'byscratch/data/adult_stratified_income.csv' > cols.txt ;
# !cat cols.txt

with open("cols.txt") as f:
    for line in f:
        cols = line.strip().split(',')

print("| i | col |")
print("|-- |---- |")
for i,c in enumerate(cols):
  print(f"| {i} | {c} |")

# Get the shape of this dataset
filename = "byscratch/data/adult_stratified_income.csv"
with open(filename) as f:
    numlines = sum(1 for line in f)

numlines

# Load up a large data structure
data_dict = csv.DictReader(open(filename))

# Let's do some frequency counting
from collections import Counter

# race, gender, occupation, hours worked per week, and education
ages = []
races = []
genders = []
incomes = []
educationalnum = []
occupations = []
hoursperweek = []

for row in data_dict:
    ags = int(row["age"])
    ages.append(ags)

    hours = int(row["hours-per-week"])
    hoursperweek.append(hours)

    race = str(row["race"])
    races.append(race)

    gender = str(row["gender"])
    genders.append(gender)

    income = str(row["income"])
    incomes.append(income)

    occupation = str(row["occupation"])
    occupations.append(occupation)

    education = str(row["educational-num"])
    educationalnum.append(int(education))

assert len(educationalnum) == len(hoursperweek) == len(races) == len(genders) == len(incomes) == len(occupations)

# SANITY CHECK Use your code to get the mean & median hours per week
print(f"The mean hours-per-week is {mean(hoursperweek)}, median is {median(hoursperweek)}")

print(f"The mean age is {mean(ages)}, median is {median(ages)}")

print(f"The mean work hours per week is {mean(hoursperweek)}, median is {median(hoursperweek)}")


# Create dictionaries of frequency counts for ordinals. 
# Order the keys

def orderedCounter(alist):
    c = Counter(alist)
    ld = dict((str(k).lower(), v) for k, v in c.items())
    old = OrderedDict(sorted(ld.items()))
    return old

def orderedCounterInt(alist):
    c = Counter(alist)
    ld = dict((int(k), v) for k, v in c.items())
    s = sorted(ld.items())
    # print(s)
    old = OrderedDict(sorted(s))
    return old

# distribution of incomes? 
inc = orderedCounter(incomes)

sns.barplot(
    y=list(inc.keys()),
    x=list(inc.values()),
    orient='h')

plt.title("We sampled the dataset so there are equal numbers of each group")
plt.show()


# Ages histplot

sns.histplot(ages,binwidth=5)

plt.title("Histogram plot showing the distribution of ages, binwidth = 5yrs")
plt.show()

# distribution of education?
edu = orderedCounterInt(educationalnum)

sns.barplot(
    y=list(edu.keys()),
    x=list(edu.values()),
    orient='h')

plt.title("Number of adults vs. years of education completed")
plt.show()

o = orderedCounter(occupations)

sns.barplot(
    y=list(o.keys()),
    x=list(o.values()),
    orient='h')

plt.title("Number of adults employed in each job type")
plt.show()

# Distribution of race?
r = orderedCounter(races)

sns.barplot(
    y=list(r.keys()),
    x=list(r.values()),
    orient='h')

plt.title("Distribution of race")
plt.show()

# distribution of gender?
g = orderedCounter(genders)

sns.barplot(
    y=list(g.keys()),
    x=list(g.values()),
    orient='h')

plt.title("Distribution of gender")
plt.show()

def col2index(d,m):
    # Python 3.6+ retains the order of dictionaries
    i=0
    for k,v in d.items():
        if k.lower() == m.lower():
            return i
        else:
            i = i+1

vec = []
X = []  # vector
y = []  # target
# encode the ordinals
for j in range(len(hoursperweek)):
    # Arrrggggh! We want the index, not the value!
    k=races[j]
    race = col2index(r,k)
    k=genders[j]
    gender = col2index(g,k)
    k=occupations[j]
    occupation = col2index(o,k)
    k=incomes[j]
    income=col2index(inc,k)
    education=educationalnum[j]
    age=ages[j]

    
    vec.append( (income,[age,gender,race,education,occupation]) )

import random
from byscratch.machine_learning import split_data

train,test = split_data(vec,0.75)
len(test),len(train)


y_train = [t[0] for t in train]
X_train = [t[1] for t in train]

y_test = [t[0] for t in test]
X_test = [t[1] for t in test]

# train should have more items, right?
len(X_train),len(X_test)


# What is the best value for k?'
# Try the elbow method
import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier

numpts = 25
error_rate = []
error_rate1 = []

# This might take some time!
for i in range(1,numpts):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train,y_train)
  pred_i = knn.predict(X_test)
  error_rate.append(np.mean(pred_i != y_test))
  pred_j = knn.predict(X_train)
  error_rate1.append(np.mean(pred_j != y_train))

plt.figure(figsize=(10,6))
plt.plot(range(1,numpts),error_rate,color='blue', 
         linestyle='dashed', marker='o',markerfacecolor='red', markersize=7)
plt.plot(range(1,numpts),error_rate1,color='gray', 
         linestyle='dashed', marker='^',markerfacecolor='green', markersize=7)

plt.legend(['testing data','training data'])
plt.title('Error Rate vs. value of k')
plt.xlabel('k value')
plt.ylabel('Error Rate')


plt.show()

k = 17

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

knn.score(X_train,y_train), knn.score(X_test,y_test)

# Plot non-normalized confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        knn,
        X_test,
        y_test,
        display_labels=['<50k','>50k'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    # print(disp.confusion_matrix)

plt.show()

sklearn.__version__

##Pandas Bayesian analysis

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

df.columns

# I cleaned out the Null values before we started
df.isnull().sum()

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# As above -- no Nulls expected
df[categorical].isnull().sum()


# Let's drop the education and fnlwgt columns
df=df.drop(['education','fnlwgt'],axis=1)

# the data
X = df.drop(['income'], axis=1)

# the target
y = df['income']

# convert ordinal data (or "features" !) to numbers
# compare the methods above

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['workclass'] = le.fit_transform(df['workclass'])
df['marital_status'] = le.fit_transform(df['marital_status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native_country'] = le.fit_transform(df['native_country'])
df['income'] = le.fit_transform(df['income'])

df.head()

sns.barplot(x='income',y='age',data=df);

sns.barplot(x='relationship',y='race',data=df);

# what two features correlate the most? least?
sns.heatmap(df.corr())

X=df.drop(['income'],axis=1)
y=df['income']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

# train should have more items, right?
len(X_train),len(X_test)

from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_pred = gnb.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)

# Plot non-normalized confusion matrix
# from sklearn.metrics import ConfusionMatrixDisplay, classification_report

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        gnb,
        X_test,
        y_test,
        display_labels=['<50k','>50k'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    # print(disp.confusion_matrix)

plt.show()

