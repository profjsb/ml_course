# from https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class BasicTransformer(BaseEstimator):
    
    def __init__(self, cat_threshold=None, num_strategy='median',
                 return_df=False):
        # store parameters as public attributes
        self.cat_threshold = cat_threshold
        
        if num_strategy not in ['mean', 'median']:
            raise ValueError('num_strategy must be either "mean" or "median"')
        self.num_strategy = num_strategy
        self.return_df = return_df
        
    def fit(self, X, y=None):
        # Assumes X is a DataFrame
        self._columns = X.columns.values
        
        # Split data into categorical and numeric
        self._dtypes = X.dtypes.values
        self._kinds = np.array([dt.kind for dt in X.dtypes])
        self._column_dtypes = {}
        is_cat = self._kinds == 'O'
        self._column_dtypes['cat'] = self._columns[is_cat]
        self._column_dtypes['num'] = self._columns[~is_cat]
        self._feature_names = self._column_dtypes['num']
        
        # Create a dictionary mapping categorical column to unique 
        # values above threshold
        self._cat_cols = {}
        for col in self._column_dtypes['cat']:
            vc = X[col].value_counts()
            if self.cat_threshold is not None:
                vc = vc[vc > self.cat_threshold]
            vals = vc.index.values
            self._cat_cols[col] = vals
            self._feature_names = np.append(self._feature_names, col 
                                            + '_' + vals)
            
        # get total number of new categorical columns    
        self._total_cat_cols = sum([len(v) for col, v in 
                                    self._cat_cols.items()])
        
        # get mean or median
        num_cols = self._column_dtypes['num']
        self._num_fill = X[num_cols].agg(self.num_strategy)
        return self
        
    def transform(self, X):
        # check that we have a DataFrame with same column names as 
        # the one we fit
        if set(self._columns) != set(X.columns):
            raise ValueError('Passed DataFrame has different columns than fit DataFrame')
        elif len(self._columns) != len(X.columns):
            raise ValueError('Passed DataFrame has different number of columns than fit DataFrame')
            
        # fill missing values
        num_cols = self._column_dtypes['num']
        X_num = X[num_cols].fillna(self._num_fill)
        
        # Standardize numerics
        std = X_num.std()
        X_num = (X_num - X_num.mean()) / std
        zero_std = np.where(std == 0)[0]
        
        # If there is 0 standard deviation, then all values are the 
        # same. Set them to 0.
        if len(zero_std) > 0:
            X_num.iloc[:, zero_std] = 0
        X_num = X_num.values
        
        # create separate array for new encoded categoricals
        X_cat = np.empty((len(X), self._total_cat_cols), 
                         dtype='int')
        i = 0
        for col in self._column_dtypes['cat']:
            vals = self._cat_cols[col]
            for val in vals:
                X_cat[:, i] = X[col] == val
                i += 1
                
        # concatenate transformed numeric and categorical arrays
        data = np.column_stack((X_num, X_cat))
        
        # return either a DataFrame or an array
        if self.return_df:
            return pd.DataFrame(data=data, 
                                columns=self._feature_names)
        else:
            return data
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def get_feature_names():
        return self._feature_names