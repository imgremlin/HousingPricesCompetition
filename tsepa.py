import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')
test_raw=test.copy()

num_col=[]
cat_col=[]


def cols_type():

    for col in train.columns:
        if train[col].dtype=='object':
            cat_col.append(col)
        else:
            num_col.append(col)
    
    #print('numerical cols:', num_col)
    #print('\nnum cat cols:', len(cat_col)) 
    #print('cat cols:', cat_col)
    return cat_col    

train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],
           axis=1, inplace=True)
test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],
           axis=1, inplace=True)
#print(train.info())    

def mapping():
    map_list=['HeatingQC','GarageCond','GarageQual','ExterQual','ExterCond',
              'BsmtQual','BsmtCond','KitchenQual']
    
    dict_qc={'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    
    for lst in map_list:
        train[lst] = train[lst].map(dict_qc)
    
    dict_ut={'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1}
    train['Utilities'] = train['Utilities'].map(dict_ut)
    dict_exp={'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0}
    train['BsmtExposure'] = train['BsmtExposure'].map(dict_exp)
    dict_fin={'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0}
    train['BsmtFinType1'] = train['BsmtFinType1'].map(dict_fin)
    train['BsmtFinType2'] = train['BsmtFinType2'].map(dict_fin)
    dict_air={'Y':1, 'N':0}
    train['CentralAir'] = train['CentralAir'].map(dict_air)
    dict_gar={'Finished':3, 'RFn':2, 'Unf':1, 'NA':0}
    train['GarageFinish'] = train['GarageFinish'].map(dict_gar)
    return train

#mapping()
   
def bar_dependence(column):
    col = train.groupby(column)['SalePrice'].mean()            
    plt.bar(col.index, col.values)     	
    plt.show()
    
cat_col_stat=['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',
              'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
              'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
              'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
              'Electrical', 'Functional', 'GarageType', 'PavedDrive',
              'SaleType', 'SaleCondition']

#for col in cat_col_stat:  
#    bar_dependence(col) 

import category_encoders as ce
cols_type()
target_enc = ce.CatBoostEncoder(cols=cat_col)
target_enc.fit(train[cat_col], train['SalePrice'])

train = train.join(target_enc.transform(train[cat_col]).add_suffix('_cb'))
test = test.join(target_enc.transform(test[cat_col]).add_suffix('_cb'))

train.drop(cat_col, axis=1, inplace=True)
test.drop(cat_col, axis=1, inplace=True)

train=train.interpolate(method='polynomial', order=2)
test=test.interpolate(method='polynomial', order=2)

#train = train.sample(frac=1).reset_index(drop=True)

from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

train=train.dropna()

from sklearn.preprocessing import MinMaxScaler
import numpy as np

#scaling data
scaler_X = MinMaxScaler(feature_range = (0, 1))
scaler_y = MinMaxScaler(feature_range = (0, 1))

#setting X ant y to prepare test/train set
X = train.drop(labels=['SalePrice'], axis=1)
y = train['SalePrice']



#feature_cols=list(X.columns)

    
columns=['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
         'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
         'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF',
         'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
         'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
         'GarageYrBlt', 'WoodDeckSF',
         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
         'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'MSZoning_cb', 'Street_cb',
         'Utilities_cb', 'LotConfig_cb',
         'LandSlope_cb', 'Neighborhood_cb', 'Condition1_cb', 'Condition2_cb',
         'HouseStyle_cb', 'RoofStyle_cb', 'RoofMatl_cb',
         'ExterQual_cb',
         'ExterCond_cb', 'BsmtQual_cb', 'BsmtCond_cb',
         'BsmtExposure_cb', 'BsmtFinType1_cb', 'Heating_cb',
         'HeatingQC_cb', 'CentralAir_cb', 'Electrical_cb', 'KitchenQual_cb',
         'Functional_cb', 'GarageFinish_cb', 
         'PavedDrive_cb', 'LotShape_cb', 
         'GarageType_cb', 'HalfBath', 'MasVnrType_cb', 'Foundation_cb',
         'Exterior1st_cb', 'SaleCondition_cb', 'TotalBsmtSF', 
         'GarageQual_cb','BsmtFinType2_cb','LandContour_cb', 
         'KitchenAbvGr', 'BldgType_cb', 'GarageCars', 'GarageArea',
         ]
dropped = ['Exterior2nd_cb', 'SaleType_cb', '1stFlrSF', 'MSSubClass',
           'GarageCond_cb',  
            
           ]    

test=test[columns]
X=X[columns]
    
train_size = 0.75
separator = round(len(X.index)*train_size)

#making train/test set
X_train, y_train = X.iloc[0:separator], y.iloc[0:separator]
X_test, y_test = X.iloc[separator:], y.iloc[separator:]

#setting our model
param_test = {
    'learning_rate': [0.01],
    'n_estimators': [750],
    'max_depth': [3], 
    'min_child_weight': [3],
    'gamma':[0],
    'colsample_bytree': [0.7],
    'subsample': [0.9],
    'reg_alpha':[0.001]
}
model = XGBRegressor(objective ='reg:squarederror',colsample_bytree= 0.7,
                     gamma= 0, learning_rate= 0.07,
 max_depth= 3, min_child_weight= 3, n_estimators= 750,
 reg_alpha= 0.001, subsample= 0.9)

def feat_plot(model):
    #feature importance plot
    feat_plot = plot_importance(model, height=2)
    feat_plot.figure.set_size_inches(10, 15)

import seaborn as sns

def corr_matrix(matrix):
    
    #correlation matrix
    corr = matrix.corr()
    plt.subplots(figsize=(25,25))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(150, 275, s=80, l=55, n=8), 
                annot=False)
    plt.title('Correlation Heatmap of Numeric Features')
    
'''    
from sklearn.feature_selection import SelectKBest, f_classif

# Keep 5 features
selector = SelectKBest(f_classif, k=30)

X_train_new = selector.fit_transform(X_train, y_train)

selected_features = pd.DataFrame(selector.inverse_transform(X_train_new), 
                                 index=X_train.index, 
                                 columns=columns)
#print(selected_features.columns)
#print(selected_features.head())
'''


from sklearn.model_selection import GridSearchCV


'''
model1_grid = GridSearchCV(model, 
                        param_test, 
                        cv=5,
                        n_jobs = 5,
                        verbose = True)
model1_grid.fit(X_train, y_train)
print(model1_grid.best_score_)
print(model1_grid.best_params_)

'''
#fitting model
model.fit(X_train, y_train,
       eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)   

#making predictions
preds = model.predict(X_test) 
predictions_test = model.predict(test) 

print('mae: {}'.format(mean_absolute_error(y_test, preds).round(1)))

#feat_plot(model)
#corr_matrix(X)

cv_results = cross_validate(model, X, y, cv=4,
                            scoring='neg_mean_absolute_error')
#print('cross val mae:', cv_results['test_score'])
print('avg cross val mae:', -1*cv_results['test_score'].mean().round(1))

# Save test predictions to file
output = pd.DataFrame({'Id': test_raw.index,
                       'SalePrice': predictions_test})

#output.to_csv('submission.csv', index=False)
