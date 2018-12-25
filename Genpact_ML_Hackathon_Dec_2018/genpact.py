#
# Analytics Vidhya - Genpact ML hackathon Dec 2018
# 5th place solution - varunbpatil
#
import pandas as pd
import numpy as np
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# Read training and test datasets.
df_train = pd.read_csv('train_GzS76OK/train.csv')
df_center_info = pd.read_csv('train_GzS76OK/fulfilment_center_info.csv')
df_meal_info = pd.read_csv('train_GzS76OK/meal_info.csv')
df_test = pd.read_csv('test_QoiMO9B.csv')


# Merge the training data with the branch and meal information.
df_train = pd.merge(df_train, df_center_info,
                    how="left",
                    left_on='center_id',
                    right_on='center_id')

df_train = pd.merge(df_train, df_meal_info,
                    how='left',
                    left_on='meal_id',
                    right_on='meal_id')


# Merge the test data with the branch and meal information.
df_test = pd.merge(df_test, df_center_info,
                   how="left",
                   left_on='center_id',
                   right_on='center_id')

df_test = pd.merge(df_test, df_meal_info,
                   how='left',
                   left_on='meal_id',
                   right_on='meal_id')


# Convert 'city_code' and 'region_code' into a single feature - 'city_region'.
df_train['city_region'] = \
        df_train['city_code'].astype('str') + '_' + \
        df_train['region_code'].astype('str')

df_test['city_region'] = \
        df_test['city_code'].astype('str') + '_' + \
        df_test['region_code'].astype('str')


# Label encode categorical columns for use in LightGBM.
label_encode_columns = ['center_id', 
                        'meal_id', 
                        'city_code', 
                        'region_code', 
                        'city_region', 
                        'center_type', 
                        'category', 
                        'cuisine']

le = preprocessing.LabelEncoder()

for col in label_encode_columns:
    le.fit(df_train[col])
    df_train[col + '_encoded'] = le.transform(df_train[col])
    df_test[col + '_encoded'] = le.transform(df_test[col])


# Feature engineering - treat 'week' as a cyclic feature.
# Encode it using sine and cosine transform.
df_train['week_sin'] = \
        np.sin(2 * np.pi * df_train['week'] / 52.143)
df_train['week_cos'] = \
        np.cos(2 * np.pi * df_train['week'] / 52.143)

df_test['week_sin'] = \
        np.sin(2 * np.pi * df_test['week'] / 52.143)
df_test['week_cos'] = \
        np.cos(2 * np.pi * df_test['week'] / 52.143)


# Feature engineering - percent difference between base price and checkout price.
df_train['price_diff_percent'] = \
        (df_train['base_price'] - df_train['checkout_price']) / \
        df_train['base_price']

df_test['price_diff_percent'] = \
        (df_test['base_price'] - df_test['checkout_price']) / \
        df_test['base_price']


# Convert email and homepage features into a single feature - 'email_plus_homepage'.
df_train['email_plus_homepage'] = \
        df_train['emailer_for_promotion'] + \
        df_train['homepage_featured']

df_test['email_plus_homepage'] = \
        df_test['emailer_for_promotion'] + \
        df_test['homepage_featured']


# Prepare a list of columns to train on.
# Also decide which features to treat as numeric and which features to treat
# as categorical.
columns_to_train = ['week',
                    'week_sin',
                    'week_cos',
                    'checkout_price',
                    'base_price',
                    'price_diff_percent',
                    'email_plus_homepage',
                    'city_region_encoded',
                    'center_type_encoded',
                    'op_area',
                    'category_encoded',
                    'cuisine_encoded',
                    'center_id_encoded',
                    'meal_id_encoded']

categorical_columns = ['email_plus_homepage',
                       'city_region_encoded',
                       'center_type_encoded',
                       'category_encoded',
                       'cuisine_encoded',
                       'center_id_encoded',
                       'meal_id_encoded']

numerical_columns = [col for col in columns_to_train if col not in categorical_columns]


# Log transform the target variable - num_orders.
df_train['num_orders_log1p'] = np.log1p(df_train['num_orders'])


# Train-Test split.
X = df_train[categorical_columns + numerical_columns]
y = df_train['num_orders_log1p']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.02, 
                                                    shuffle=False)


# Train the LightGBM model on the best parameters obtained by grid search.
g = {'colsample_bytree': 0.4,
     'min_child_samples': 5,
     'num_leaves': 255}

estimator = LGBMRegressor(learning_rate=0.003,
                          n_estimators=40000,
                          silent=False,
                          **g)

fit_params = {'early_stopping_rounds': 1000,
              'feature_name': categorical_columns + numerical_columns,
              'categorical_feature': categorical_columns,
              'eval_set': [(X_train, y_train), (X_test, y_test)]}

estimator.fit(X_train, y_train, **fit_params)


# Get predictions on the test set and prepare submission file.
X = df_test[categorical_columns + numerical_columns]

pred = estimator.predict(X)
pred = np.expm1(pred)

submission_df = df_test.copy()
submission_df['num_orders'] = pred
submission_df = submission_df[['id', 'num_orders']]
submission_df.to_csv('submission.csv', index=False)
