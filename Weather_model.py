#!/usr/bin/env python
# coding: utf-8

# #### Import the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime


# #### Read the csv file

# In[2]:


pd.set_option('display.max_columns',None)
df=pd.read_csv('weatherAUS.csv')
df.head()


# #### Lets check for null values

# In[3]:


fig,ax=plt.subplots(figsize=(12,8))
ax=sns.heatmap(df.isna(),cbar=False)
ax.set_xlabel('Hours', fontsize=10)
ax.set_ylabel('Day', fontsize=10)
fig.tight_layout()


# ###### Clearly a few columns have significantly large amount of null values
# 
# ###### Lets find out which ones

# In[4]:


miss_df=round(((df.isna().sum())/df.isna().count())*100,2).sort_values(ascending=False).head(4).reset_index()
fig,ax=plt.subplots(figsize=(10,6))
bars=ax.bar(x=miss_df['index'],height=miss_df[0],color='crimson')
ax.tick_params(bottom=False,left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
for bar in bars:
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.3,
      round(bar.get_height(), 1),
      color='crimson',
      horizontalalignment='center',
      weight='bold'
  )
plt.title('Top 4 Columns with missing values',fontsize=15)

fig.tight_layout()


# ###### While these are huge number of null values they are still less than 50% and thus we will use them by mutating them 

# In[5]:


df.select_dtypes(include=['object']).columns


# In[6]:


def replace_missing(data,col):
    df[col] = df[col].fillna(df[col].mode()[0])
    return df

replace_missing(df,'WindGustDir')
replace_missing(df,'WindDir9am')
replace_missing(df,'WindDir3pm')
replace_missing(df,'RainToday')
replace_missing(df,'RainTomorrow')


# ##### Now we have to deal with categorical variables

# In[7]:


for col in df.select_dtypes(include='O'):
    print(f'{col}: {df[col].unique()}')


# ##### Lets change these values from Categorical to Numerical

# In[8]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
cat_var=[features for features in df.select_dtypes(include='O')]
for col in cat_var:
    df[col]=enc.fit_transform(df[col])


# In[9]:


df.head()


# #### Now that, that's been taken care of lets move to the missing values

# In[10]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

MiceImputed = df.copy(deep=True) 
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(df)


# ##### Check if the data is balnced or not

# In[11]:


MiceImputed['RainTomorrow']=MiceImputed['RainTomorrow'].astype('int')
MiceImputed['RainToday']=MiceImputed['RainToday'].astype('int')
MiceImputed['WindGustDir']=MiceImputed['WindGustDir'].astype('int')
MiceImputed['WindDir9am']=MiceImputed['WindDir9am'].astype('int')
MiceImputed['WindDir3pm']=MiceImputed['WindDir3pm'].astype('int')
MiceImputed['Location']=MiceImputed['Location'].astype('int')


# In[12]:


MiceImputed.head()


# #### Now check the relationship between the variables

# In[13]:


mask=np.triu(MiceImputed.corr())
fig, ax = plt.subplots(figsize=(20, 20))
ax=sns.heatmap(df.corr(),annot=True,mask=mask)
fig.tight_layout()


# Some variables are showing high correlations among themselves but the corelation is not exactly equal to 1 so we can keep the columns. 
# These are pressure and temperature variables. so we will keep an eye out for them

# ##### For now lets look at the relations between these variables using pairplot

# In[14]:


sns.pairplot(MiceImputed,vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm', 'Temp9am', 'Temp3pm'), hue='RainTomorrow')


# In[15]:


MiceImputed['RainTomorrow'].value_counts()


# ##### Next we check if the data is imbalanced

# In[16]:


fig,ax=plt.subplots(figsize=(12,6))
ax=MiceImputed['RainTomorrow'].value_counts().plot(kind='bar')
plt.xlabel('Rain tommorow')
plt.ylabel(' Value Count')
plt.show()


# In[17]:


round(MiceImputed['RainTomorrow'].value_counts()/MiceImputed['RainTomorrow'].count()*100,2)


# ##### As we can see the data is highly imbalaced with ratio of 77:22. Instead of undersampling we willl go for oversampling the minority class with SMOTEEEN 
# 
# 
# 
# 
# ###### But first lets plit the data into X and Y

# In[18]:


X=MiceImputed.drop('RainTomorrow',axis=1)
y=MiceImputed.RainTomorrow


# In[19]:


from imblearn.combine import SMOTEENN
sm=SMOTEENN()
X_sm,y_sm=sm.fit_resample(X,y)


# ##### Featue selection Time, lets use a wrapper method to do this

# In[20]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(n_estimators=100,random_state=42)
model.fit(X_sm,y_sm)
features=pd.Series(model.feature_importances_,index=X_sm.columns)
features.nlargest(20).plot(kind='barh')
plt.show()
# support = select.get_support()
# features = X_sm.loc[:,support].columns.tolist()
# # print(features)
# print(ExtraTreesClassifier(n_estimators=100, random_state=0).fit(X_sm,y_sm).feature_importances_)


# In[21]:


X_sm.drop(['Date'],axis=1,inplace=True)


# ##### Ok now its time for train test split

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.33, random_state=42)


# ##### Normalize the data. we will have outliers and so we wil use Robust Scaler

# In[23]:


from sklearn.preprocessing import RobustScaler
ro=RobustScaler()
X_train=ro.fit_transform(X_train)
X_test=ro.transform(X_test)


# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score


def make_classification(X_train, X_test, y_train, y_test):
    accuracy, f1 = [], []

    random_state = 42

    ##classifiers
    classifiers = []
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(LogisticRegression(random_state=random_state, solver='lbfgs', max_iter=10000))
    classifiers.append(XGBClassifier(random_state = random_state))
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(KNeighborsClassifier())

    for classifier in classifiers:
        # classifier and fitting
        clf = classifier
        clf.fit(X_train, y_train)

        # predictions
        y_preds = clf.predict(X_test)

        # metrics
        accuracy.append(((accuracy_score(y_test, y_preds))) * 100)
        f1.append(((f1_score(y_test, y_preds))) * 100)

    results_df = pd.DataFrame({"Accuracy Score": accuracy,
                               "f1 Score": f1,
                               "ML Models": ["DecisionTree", "AdaBoost",
                                             "RandomForest", "GradientBoosting",
                                             "KNeighboors",'XGB',
                                             "SVC", "KNN"]})

    results = (results_df.sort_values(by=['f1 Score'], ascending=False)
               .reset_index(drop=True))

    return classifiers, results


classifiers, results = make_classification(X_train, X_test, y_train, y_test)

results


# In[27]:


model = AdaBoostClassifier()
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)


# In[28]:


from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_predicted))


# In[35]:


fig,ax = plt.subplots(figsize=(10,6))
ax = sns.heatmap(confusion_matrix(y_test,y_predicted),annot=True,fmt='d')
plt.xlabel('Truth')
plt.ylabel('Prediction')

fig.tight_layout()


# In[36]:


import pickle
pickle.dump(model,open('Weathermodel.pkl','wb'))


# In[ ]:




