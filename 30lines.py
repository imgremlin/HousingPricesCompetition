import pandas as pd
import numpy as np
train = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')
test_raw=test.copy()  
train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1, inplace=True)
test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1, inplace=True)
import category_encoders as ce
cat_cols=list(train.select_dtypes(include=['object']).columns)
target_enc = ce.CatBoostEncoder(cols=cat_cols)
target_enc.fit(train[cat_cols], train['SalePrice'])
train = train.join(target_enc.transform(train[cat_cols]).add_suffix('_cb'))
test = test.join(target_enc.transform(test[cat_cols]).add_suffix('_cb'))
train.drop(cat_cols, axis=1, inplace=True)
test.drop(cat_cols, axis=1, inplace=True)
train=train.interpolate(method='polynomial', order=2)
test=test.interpolate(method='polynomial', order=2)
train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
X = train.drop(labels=['SalePrice'], axis=1)
y = train['SalePrice']
from catboost import CatBoostRegressor
catb = CatBoostRegressor(verbose=False,learning_rate=0.01,n_estimators=3000)
catb.fit(X, y)
catb_pred = catb.predict(test)
output = pd.DataFrame({'Id': test.index, 'SalePrice': catb_pred})
output.to_csv('submission.csv', index=False)
