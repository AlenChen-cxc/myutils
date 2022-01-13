from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer

def preprocessing_pipe():

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median', )),
        ('scaler', RobustScaler())
        ])


    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

    num_col = make_column_selector(dtype_include=['float64', 'int64'])
    cat_col = make_column_selector(dtype_include=['object','category'])

    preprocessor = ColumnTransformer([('num_tr', num_transformer, num_col),
                                      ('cat_tr', cat_transformer, cat_col)])

    return preprocessor
