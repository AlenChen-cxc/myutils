from myutils.preprocessing_pipe import preprocessing_pipe
from sklearn.compose import _column_transformer

def test_preprocessing_pipe():
    assert type(preprocessing_pipe()) == _column_transformer.ColumnTransformer
