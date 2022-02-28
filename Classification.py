import csv
import time
import json
import shap
import PATHS
import random
import pickle
import Analysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, precision_score ,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from yellowbrick.classifier import ClassificationReport

# Module for model training and comparison class.

# To Do:
# 1.Expand the concept of training binary classificators beyond just fraud -
#   make Analysis.py compatible with generating feature sets for Gambling/Exchanges/Pools classification as well.


class Trainer:
    """
    Class for training and evaluation multiple models.
    """
    def __init__(self,random_state,fraud=False):
        self.fraud = False
        self.random_state = random_state

        if fraud:
            self.fraud =True
            self.raw_data = Analysis.get_feature_book(for_ml=True,fraud=True)
        else:
            self.raw_data = Analysis.get_feature_book(for_ml=True)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split_train_test()

        self.models = []

        self.results = dict()


    def split_train_test(self):

        splitter = StratifiedShuffleSplit(test_size=.2, n_splits=1, random_state=self.random_state)
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


    def load_model(self,fname):
        return pickle.load(open(PATHS.ML_PATH  + fname, 'rb'))


    def log(self):
        with open(PATHS.ML_PATH+'log.csv','a') as f:
            writer = csv.writer(f)
            for key, val in self.results.items():
                writer.writerow([key,
                                 val[1],
                                 val[2],
                                 val[0]])

    def shap(self, chosen_models):
        """
        plots a shap value plot.
        """
        for model in chosen_models:
            shap_values = shap.TreeExplainer(model).shap_values(self.X_train)
            shap.summary_plot(shap_values, self.X_train, plot_type="bar")

    def report(self,model,save=False):
        """
        plots and saves a confusion matrix.
        """
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


def multiple_evaluations(n,model):
    """
    Runs n training cycles of the chosen model on random splits.
    Saves precision/recall plots.
    """
    name = model.__str__().split('(')[0]
    precision_recall_list = []
    for i in range(n):
        trainer_for_avg = Trainer(random.randint(0, 1000), fraud=True)
        trainer_for_avg.models.append(model)
        trainer_for_avg.models[0].fit(trainer_for_avg.X_train, trainer_for_avg.Y_train)
        pred = trainer_for_avg.models[0].predict(trainer_for_avg.X_test)
        precision_recall_list.append(
            (precision_score(trainer_for_avg.Y_test, pred), recall_score(trainer_for_avg.Y_test, pred)))

    plt.close()
    plt.hist([prec[0] for prec in precision_recall_list], bins=20)
    plt.suptitle('Histogram of precision score')
    plt.title(f'{name}, {n} runs')
    plt.xlabel('Precision score')
    plt.ylabel('Count')
    plt.savefig(f'{name}_/mnt/plots/precision_hist2_{n}.png')
    plt.show()
    plt.close()
    plt.hist([prec[1] for prec in precision_recall_list], bins=20,color='purple')
    plt.suptitle('Histogram of recall score')
    plt.title(f'{name}, {n} runs')
    plt.xlabel('Recall score')
    plt.ylabel('Count')
    plt.savefig(f'{name}_/mnt/plots/recall_hist_{n}.png')
    plt.show()
    plt.close()
    plt.scatter([prec[0] for prec in precision_recall_list],[prec[1] for prec in precision_recall_list],c=[prec[0]*prec[1] for prec in precision_recall_list],cmap='plasma')
    plt.suptitle('Precison/Recall Score')
    plt.title(f'{name}, {n} runs')
    plt.xlabel('Precision score')
    plt.ylabel('Recall score')
    plt.savefig(f'/mnt/plots/'
                f'{name}_precision_recall_scatter_{n}.png')
    plt.show()