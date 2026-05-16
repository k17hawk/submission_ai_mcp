from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class QuestionMarkToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy() 
        X = X.replace('?', np.nan)
        return X
    