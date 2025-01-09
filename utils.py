
import os
from joblib import PrintTime
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler, SMOTENC

DATA_DIR = './data'
RANDOM_SEED = 42

def read_data():
    data = pd.read_csv(os.path.join(DATA_DIR, 'healthcare-dataset-stroke-data.csv'))
    return data

def preprocess(data: pd.DataFrame):
    """fill the missing values of bmi
        implementing standardization to continuous variables
        implement one-hot encoding to categorical varirables

    Args:
        data (pd.DataFrame): original data

    Returns:
        pd.DataFrame: processes data
    """
    data = data.drop('id', axis=1)
    var_int = data.dtypes[data.dtypes == 'int'].index.tolist()
    data[var_int] = data[var_int].astype('object')
    var_cat = data.dtypes[data.dtypes == 'object'].index.tolist()
    var_float = data.dtypes[data.dtypes == 'float'].index.tolist()
    le = LabelEncoder()
    data_cat = data[var_cat]
    data[var_cat] = data[var_cat].apply(le.fit_transform)

    column_names = data.columns.tolist()
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    data = imputer.fit_transform(data)
    data = pd.DataFrame(data, columns=column_names)
    data[var_cat] = data_cat.values

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), var_float),
            ('oe', OneHotEncoder(sparse_output=False, drop='first'), var_cat)
        ])
    data = preprocessor.fit_transform(data)
    column_names = np.concat([preprocessor.transformers_[i][1].get_feature_names_out() for i in range(2)]).tolist()
    data = pd.DataFrame(data, columns=column_names)
    data = data.rename(columns={'stroke_1': 'stroke'})
    return data


def preprocess4r(data: pd.DataFrame):
    """fill the missing values of bmi
        implementing standardization to continuous variables
        implement one-hot encoding to categorical varirables

    Args:
        data (pd.DataFrame): original data

    Returns:
        pd.DataFrame: processes data
    """
    data = data.drop('id', axis=1)
    data = data[data['gender'] != 'Other']
    var_int = data.dtypes[data.dtypes == 'int'].index.tolist()
    data[var_int] = data[var_int].astype('object')
    var_cat = data.dtypes[data.dtypes == 'object'].index.tolist()
    var_float = data.dtypes[data.dtypes == 'float'].index.tolist()
    
    le = LabelEncoder()
    data_cat = data[var_cat]
    data[var_cat] = data[var_cat].apply(le.fit_transform)
    column_names = data.columns.tolist()
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    data = imputer.fit_transform(data)
    data = pd.DataFrame(data, columns=column_names)
    
    data[var_cat] = data_cat.values
    data['hypertension'] = data['hypertension'].replace({0: 'No', 1: 'Yes'})
    data['heart_disease'] = data['heart_disease'].replace({0: 'No', 1: 'Yes'})
    data['stroke'] = data['stroke'].replace({0: 'No', 1: 'Yes'})
    
    X, y = data.iloc[:, :-1], data['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, shuffle=True)
    var_cat.pop()
    oversample = SMOTENC(categorical_features=var_cat, sampling_strategy='auto', random_state=RANDOM_SEED)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    train_df = pd.concat((X_train, y_train), axis=1)
    test_df = pd.concat((X_test, y_test), axis=1)

    return train_df, test_df

def oversampling(data):
    count_not_stroke, count_stroke = data.stroke.value_counts()
    data_not_stroke = data[data["stroke"] == 0]
    data_stroke = data[data["stroke"] == 1]
    data_stroke_oversampling = data_stroke.sample(count_not_stroke, replace = True)
    data = pd.concat([data_not_stroke, data_stroke_oversampling], axis = 0)
    return data

# data = read_data()

# train, test = preprocess4r(data)
# train.to_csv('./data/train_data_4r.csv', index=False)
# test.to_csv('./data/test_data_4r.csv', index=False)

