import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')
test_raw=test.copy()

num_col=[]
cat_col=[]

#making list with numerical and categorical columns
def cols_type():
    
    for col in train.columns:
        if train[col].dtype=='object':
            cat_col.append(col)
        else:
            num_col.append(col)

    return cat_col    

#these columns have got to many missing values
train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],
           axis=1, inplace=True)
test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],
           axis=1, inplace=True)

#function for visualising numerical columns to understand their deviation
def scatter_depence(column):
    plt.figure(figsize=(10,6))
    plt.scatter(X[column], y, color='royalblue',
                s=10, alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('SalePrice')
    plt.show()
    
#function for visualising categorical columns to understand their deviation
def bar_dependence(column):
    col = train.groupby(column)['SalePrice'].mean()            
    plt.bar(col.index, col.values)     	
    plt.show()

#decoding categorical columns w/ catb enc
import category_encoders as ce
cols_type()
target_enc = ce.CatBoostEncoder(cols=cat_col)
target_enc.fit(train[cat_col], train['SalePrice'])

train = train.join(target_enc.transform(train[cat_col]).add_suffix('_cb'))
test = test.join(target_enc.transform(test[cat_col]).add_suffix('_cb'))

train.drop(cat_col, axis=1, inplace=True)
test.drop(cat_col, axis=1, inplace=True)

#handling with missing values
train=train.interpolate(method='polynomial', order=2)
test=test.interpolate(method='polynomial', order=2)

#deleting outliers
train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train[train['GarageArea'] < 1200]

from sklearn.preprocessing import RobustScaler
import numpy as np

#dividing train set on X&y
X = train.drop(labels=['SalePrice'], axis=1)
y = train['SalePrice']

cols = X.select_dtypes(np.number).columns

#scaling X w/ RobustScaler to level some outliers
 #maybe had to fit&transform X&y together
scaler=RobustScaler()
X[cols] = scaler.fit_transform(X[cols])
test[cols] = scaler.transform(test)

#log our target to make it closer to Gaussian deviatian
y=np.log(y)

import seaborn as sns

#for feature engineering
def corr_matrix(matrix):
    corr = matrix.corr()
    plt.subplots(figsize=(25,25))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(150, 275, s=80, l=55, n=8), 
                annot=False)
    plt.title('Correlation Heatmap of Numeric Features')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#preparing for cross validation
kfold = KFold(n_splits=4, random_state=0)

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

#custom grid search))
'''
n_estim = [200, 500, 1000, 3000, 5000, 10000]
learn_rate=[0.003, 0.01, 0.03, 0.06, 0.08, 0.1]

for ne in n_estim:
    for lr in learn_rate:
        lgbm = LGBMRegressor(objective='regression', verbose=-1,
                             learning_rate=lr, n_estimators=ne)
        results = cross_val_score(lgbm, X, y, cv=kfold, scoring='r2')
        print(f"ne:{ne} lr:{lr} res: {results.mean():.5f}")
'''

#different already tuned regressors
xgb = XGBRegressor(objective ='reg:squarederror',learning_rate=0.01,
                   n_estimators=3500, max_depth=3, min_child_weight=2,
                   colsample_bytree=0.55, subsample=0.65)
catb = CatBoostRegressor(verbose=False,
                         learning_rate=0.01,n_estimators=3000,
                         subsample=0.6,colsample_bylevel=0.8)
lgbm = LGBMRegressor(objective='regression', verbose=-1,
                             learning_rate=0.01, n_estimators=700,
                             colsample_bytree=0.6, subsample=0.6,
                             colsample_bylevel=0.6, colsample_bynode=0.6,
                             num_leaves=22, bagging_freq=1, bagging_seed=5)

#check separate model with cross validation
#results = cross_val_score(lgbm, X, y, cv=kfold, scoring='r2')
#print(f"res kfold: {results.mean():.5f}")

from sklearn.model_selection import train_test_split

#train split test with random state!=0 for 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size = 0.25, random_state = 5)

#custom grid search of coefficients for ensemble learning
'''
from sklearn.metrics import r2_score

#print(r2_score(y_test,lgbm_pred).round(4))

xgb_tabl=[]
catb_tabl=[]
lgbm_tabl=[]
res_tabl=[]

for xgb_i in range(11):
    for cat_i in range(11-xgb_i):
        
        lgbm_i = 10 - xgb_i - cat_i
        
        xgb_tabl.append(xgb_i)
        catb_tabl.append(cat_i)
        lgbm_tabl.append(lgbm_i)
        y_pred = 0.1 * (xgb_i*xgb_pred + cat_i*catb_pred + lgbm_i*lgbm_pred)
        res_tabl.append(r2_score(y_test,y_pred).round(6))
        #print(f"{number}. xgb={xgb/10} cat={cat/10} lgbm={lgbm/10}")

dict_table = {'xgb': xgb_tabl, 'lgbm': lgbm_tabl,
              'catb': catb_tabl, 'res':res_tabl}  

df = pd.DataFrame(dict_table)
df.sort_values(by="res", ascending=False)

df.to_csv('result_table.csv', index=False)
'''

#function to fit, predict and calculate final prediction
def fit_pred(X, y, test, xgb_i, lgbm_i, cat_i):
    
    catb.fit(X, y)
    xgb.fit(X, y)
    lgbm.fit(X, y)

    catb_pred = catb.predict(test)
    xgb_pred = xgb.predict(test)
    lgbm_pred = lgbm.predict(test)
    
    total_pred=xgb_i*xgb_pred + cat_i*catb_pred + lgbm_i*lgbm_pred
    
    return total_pred


#decoding log y with exp
output_pred=np.exp(fit_pred(X,y,test,0.2,0.1,0.7))

#making submission
output = pd.DataFrame({'Id': test_raw.index, 'SalePrice': output_pred})
output.to_csv('submission.csv', index=False)
