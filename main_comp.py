
from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics, svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier 

from bs4 import BeautifulSoup
from time import strftime, gmtime

import pandas as pd 
import xgboost as xgb
import numpy as np 
import matplotlib.pyplot as plt


np.warnings.filterwarnings('ignore')

def load_data():
    #df = pd.read_csv('data/train_and_test.csv')
    df = pd.read_csv('data/data_sample.csv')
    df = df.loc[:, ['complain', 'is complaint valid']]
    df = df.dropna(subset=['is complaint valid'])
    df['complain'] = df['complain'].apply(lambda item : BeautifulSoup(item ,"lxml").text)
    
    # return randomly shuffled dataframe
    return df.sample(frac=1)

def run(classifier, params):
    df = load_data()
    x, y = df['complain'], df['is complaint valid']
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, shuffle=False)

    #print(len(x_tr), len(y_tr))#, len(x_te), len(y_te))
    
    pipe = Pipeline([ ('vect', CountVectorizer()), 
                      ('tfidf', TfidfTransformer()),
                      ('clf', classifier) ])
    
    param_list =   {'vect__ngram_range': [(1, 2)],
                    'vect__max_df': [ 0.9, 0.95],
                    'vect__min_df': [0.05],
                    'vect__stop_words': [ "english" ] }
    param_list.update(params)

    kf = KFold( n_splits=len(x_tr) )
    gs = GridSearchCV(pipe, param_list, verbose=1, cv=10)
    
    gs.fit(x_tr, y_tr)

    print("\nBest score: {}".format( gs.best_score_ ))
    print("\nBest estimator: {}".format( gs.best_estimator_ ))
    print("\nBest parameters: {}".format( gs.best_params_ ))
    
    y_pred = gs.predict(x_te)
    
    eval(y_te, y_pred)

    #plot_(y_te, y_pred)

def plot_(y_true, y_pred):
    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
    ax.plot( [min(y_true), max(y_true)], [min(y_pred)], max(y_pred), lw=2 )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    
    plt.show()


def eval(targ, pred):
    print("\n\tF1: {}\n\n".format(metrics.f1_score(targ, pred, pos_label='Y')))


def regression():
    params = {'clf__penalty': ['l2'],
              'clf__C': [1e-3, 1e-2, 1e-1, 1 ],
              'clf__class_weight': [ "balanced" ] }
    clf = linear_model.LogisticRegression()

    run(clf, params)

def boosted_tree():
    params = {'clf__max_depth': [5, 6, 7],
                  'clf__n_estimators': [15, 18, 20],
                  #'clf__silent': [False],
                  #'clf__learning_rate': [1e-1, 1e-2, 1e-3],
                  'clf__booster': ['gbtree', 'gblinear'],
                  'clf__objective': ['reg:logistic'] }
    clf = xgb.XGBClassifier()

    run(clf, params)

def random_forest():
    params = {
        'clf__n_estimators': [8, 9, 10],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_features': [4, 5],
        'clf__max_depth': [8, 9, 10]
    }
    clf = RandomForestClassifier()

    run(clf, params)


def svm_clf():
    params = {  'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                'clf__gamma': [1e-2, 1e-1, 1, 1e1, 1e2],
                'clf__kernel': ['linear', 'rbf', 'sigmoid', 'poly']  }
    clf = svm.SVC()

    run(clf, params)


if __name__ == '__main__':

    print("\nStarted: {}".format( strftime("%Y-%m-%d %H:%M:%S", gmtime()) ))

    print("\n\tLogistic Regression: ")
    regression()

    print("\n\tBoosted Trees ")
    boosted_tree()

    print("\n\tRandom Forests ")
    random_forest()

    print("\n\tSVM: ")
    svm_clf()

    print("\nFinished: {}".format( strftime("%Y-%m-%d %H:%M:%S", gmtime()) ))
