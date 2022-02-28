import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

%matplotlib inline

df = pd.read_csv(path + 'Auto_Data_Preped.csv')
df.columns

df.skew()

df['price'].hist()

log_price = np.log1p(df.price)
log_price.hist()
print(log_price.skew())

def bin_check(y):
  if y.dtype != 'O':
    ival = y.describe().values[2]; #print(ival)
    b = []
    for i in range(0,int(max(y)),int(ival)):b.append(i); # print(b)
    y_bin = pd.cut(y, bins=b,labels=[i for i in range(len(b)-1)])
  else:
    y_bin = y 
  return y_bin

def feature_analysis(x,y):
  x=x.dropna()
  x['y']=bin_check(y)
  feature_scores={}
  for col in x.columns:
    if col == 'y': continue 
    if x[col].dtype != 'O':
      feature_scores[col]=abs(np.corrcoef(y,x[col])[0][1])

    else:
      gx = x.groupby([col,'y'])['y'].count()
      gxr = gx/len(x)
      fgx = gxr[gxr.values>0.05]

      ix = fgx.index
      clist={}
      for c, yb in ix:
        if c not in clist: clist[c] = []
        clist[c].append(yb)
      availability = len(clist)/len(x[col].unique())
      penalty=[]
      for k, v in clist.items():
        penalty.append(1/len(v))
      avg_penalty = np.mean(np.array(penalty))
      ns = availability*avg_penalty
      feature_scores[col]=ns
  return feature_scores

fs = feature_analysis(df,df.price)
fs

bin_check(df.body_style)

df.head()

from math import log, log10, log2, log1p
log(13495)

lxlog = np.log(lx)
print(lxlog)
plt.plot(lxlog)
plt.show()

np.exp(lxlog)

df.describe()

df.describe(include="object")

cc = np.corrcoef(df.log_price,df.engine_size)
print(cc)
print(cc[0][1])


ordinal_labels = {'high':3, 'medium':2, 'low':1}

print(df['body_style'].unique())
Features = df['body_style']
enc = preprocessing.LabelEncoder()
enc.fit(Features)
Features = enc.transform(Features)
print(Features)

print(Features.shape)
print(Features[:5])

Features = Features.reshape(-1,1)
print(Features.shape)
print(Features[:5])

ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features)
oheFeatures = encoded.transform(Features).toarray()
print(oheFeatures[:10, :])

#multi dimension slicing for example only
oheFeatures[:5, :3]

def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    oheFeatures = encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()
    return oheFeatures
  
 categorical_columns = ['fuel_type', 'aspiration', 'drive_wheels', 'num_of_cylinders']

for col in categorical_columns:
    print(df[col].unique())
    temp = encode_string(df[col])
    Features = np.concatenate([oheFeatures, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])    


Features = np.concatenate([Features, np.array(df[['curb_weight', 'horsepower', 'city_mpg']])], axis = 1)
Features[:2,:]


## Randomly sample cases to create independent training and test data
labels = np.array(df['price'])
x_train,x_test,y_train,y_test = ms.train_test_split(Features, labels)


print(x_train.shape)
print(x_test.shape)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
x_train[:5,:]

## define and fit the linear regression model
lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(x_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))
   
y_score = lin_mod.predict(x_test) 
print_metrics(y_test, y_score, 28)    

def hist_resids(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    
hist_resids(y_test, y_score)    

def resid_qq(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    
resid_qq(y_test, y_score)   

def resid_plot(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')

resid_plot(y_test, y_score) 

y_score_untransform = np.exp(y_score)
y_test_untransform = np.exp(y_test)
resid_plot(y_test_untransform, y_score_untransform) 














































  
  
  













































