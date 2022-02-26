import PATHS
import Analysis
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



def split_train_test(feature_df):
    splitter = GroupShuffleSplit(test_size=.15, n_splits=1, random_state = 7)
    split = splitter.split(feature_df, groups=feature_df['y'])
    train_indx,test_indx = next(split)
    return feature_df.iloc[train_indx].drop('y',axis=1), feature_df.iloc[test_indx].drop('y',axis=1), feature_df.iloc[train_indx]['y'],feature_df.iloc[test_indx]['y']



def prepare_data():
    fb = Analysis.get_feature_book(for_ml=True)
    X_train, X_test, Y_train, Y_test = split_train_test(fb)




