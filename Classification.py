import csv
import time
import json
import PATHS
import pickle
import Analysis
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from yellowbrick.classifier import ClassificationReport

# Module for model training and comparison class.

# To Do:
# 1.Expand the concept of training binary classificators beyond just fraud -
#   make Analysis.py compatible with generating feature sets for Gambling/Exchanges/Pools classification as well.


class Trainer:
    def __init__(self,fraud=False):
        self.fraud = False

        if fraud:
            self.fraud =True
            self.raw_data = Analysis.get_feature_book(for_ml=True,fraud=True)
        else:
            self.raw_data = Analysis.get_feature_book(for_ml=True)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split_train_test()

        self.models = []

        self.results = dict()


    def split_train_test(self):
        splitter = StratifiedShuffleSplit(test_size=.2, n_splits=1, random_state=7)
        if self.fraud:
            split = splitter.split(self.raw_data, self.raw_data['is_fraud'])
            train_indx, test_indx = next(split)
            return self.raw_data.iloc[train_indx].drop('is_fraud', axis=1), self.raw_data.iloc[test_indx].drop('is_fraud', axis=1), \
                   self.raw_data.iloc[train_indx]['is_fraud'], self.raw_data.iloc[test_indx]['is_fraud']
        else:
            split = splitter.split(self.raw_data, self.raw_data['y'])
            train_indx, test_indx = next(split)
            return self.raw_data.iloc[train_indx].drop('y', axis=1), self.raw_data.iloc[test_indx].drop('y', axis=1), \
                   self.raw_data.iloc[train_indx]['y'], self.raw_data.iloc[test_indx]['y']


    def train(self, chosen_models: list, save=False):
        t = time.time()
        print(f'Strating training:')
        for model in chosen_models:
            m = time.time()
            print(f'Traning {model.__str__().replace("()","")}..')
            model.fit(self.X_train,self.Y_train)
            p = model.predict(self.X_test)
            self.results[model.__str__().replace("()", "")] = [p, accuracy_score(self.Y_test, p),
                                                               (time.time() - m) / 60]
            if save:
                self.save_model(model)
            print(f'Done traning {model.__str__().replace("()","")}, Duration: {(time.time() -m)/60} minutes.')
        print(f'Finished training all, Duration: {(time.time() -t)/60} minutes.')


    def full_cycle(self,  chosen_models: list, save=False):
        self.train(chosen_models,save)
        print('Reults:')
        for model in chosen_models:
            print(f'{model.__str__().replace("()","")} '
                  f'Accuracy score: {self.results[model.__str__().replace("()","")][1]},'
                  f'Unique values: {np.unique(self.results[model.__str__().replace("()","")][0],return_counts=True)}')


    def save_model(self,model):
        fname = model.__str__().split('(')[0] +'_scr_' +str(round(self.results[model.__str__().replace("()","")][1],3))
        if self.fraud:
            fname = 'fraud_' +fname
        pickle.dump(model, open(PATHS.ML_PATH+'models/'+fname,'wb'))


    def log(self):
        with open(PATHS.ML_PATH+'log.csv','a') as f:
            writer = csv.writer(f)
            for key, val in self.results.items():
                writer.writerow([key,
                                 val[1],
                                 val[2],
                                 val[0]])


    def report(self,model,save=False):
        score = round(self.results[model.__str__().replace("()","")][1],3)
        name = model.__str__().split('(')[0]
        if self.fraud:
            vis = ClassificationReport(model,
                                       classes=["legit","fraud"],
                                       title=f"{name} Classification Report \n with accuracy {score}",
                                       support=True)
            vis.score(self.X_test, self.Y_test)
        else:
            vis = ClassificationReport(model, classes=["fraud", "exchange", "gambling", "historic", "pools", "services"],
                                       title=f"{name} Classification Report \n with accuracy {score}",
                                       support=True)
            vis.score(self.X_test, self.Y_test)
        if save:
            vis.show(f'/mnt/plots/{name}_score_matrix.png')
        vis.show()