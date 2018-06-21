"""
    This file takes care of combining the titles and complaints to generate
    more features using tfidf. Hopefully improviong the performance 

"""

from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics, svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier 
from bs4 import BeautifulSoup
from scipy.sparse import hstack

import pandas as pd 
import xgboost as xgb
import numpy as np 
from time import strftime, gmtime

np.warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data/train_and_test.csv')
    #df = pd.read_csv('data/data_sample.csv')
    df = df.loc[:, ['title', 'complain', 'is complaint valid']]
    df = df.dropna()
    df['complain'] = df['complain'].apply(lambda item : BeautifulSoup(item ,"lxml").text)
    
    # return randomly shuffled dataframe
    return df.sample(frac=1)

def run(classifier, params):
    df = load_data()

    # generate data
    comp_, title_, y = df['complain'], df['title'], df['is complaint valid']

    c_tr, c_te, t_tr, t_te, y_tr, y_te = train_test_split(comp_, title_, y, test_size=0.25, shuffle=False)

    '''
    pipe = Pipeline([ ('tfidf', TfidfVectorizer()),
                      ('clf', classifier) ])
    '''
    param_list =   {'tfidf__ngram_range': [(1, 2), (1, 3)],
                    'tfidf__max_df': [0.9, 0.95, 0.99],
                    'tfidf__min_df': [0.01, 0.05, 0.1],
                    'tfidf__stop_words': [ "english" ] }
    param_list.update(params)
    
    tfidf = TfidfVectorizer(stop_words="english",min_df=0.05,max_df= 0.95)

    # create tfidf matrices 
    title_idf = tfidf.fit_transform(t_tr)
    t_test = tfidf.transform(t_te)

    comp_idf = tfidf.fit_transform(c_tr)
    c_test = tfidf.transform(c_te)

    xtr_idf = hstack( [title_idf, comp_idf] )

    # kfold cv
    kf = KFold( n_splits=len(t_te) )

    gs = GridSearchCV(classifier, params, verbose=1, cv=kf)
    gs.fit(xtr_idf, y_tr)

    print("\nBest score: {}".format( gs.best_score_ ))
    print("\nBest estimator: {}".format( gs.best_estimator_ ))
    print("\nBest parameters: {}".format( gs.best_params_ ))
    
    xte_idf = hstack( [t_test, c_test] ) 

    y_pred = gs.predict(xte_idf)

    #print(y_pred)
    #for x, y in zip(y_pred, y_)

    eval(y_te, y_pred)


def eval(targ, pred):
    print("\n\tF1: {}\n\n".format(metrics.f1_score(targ, pred, pos_label='Y')))


def regression():
    params = {'penalty': ['l1', 'l2'],
              'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
              'class_weight': [ "balanced" ] }
    clf = linear_model.LogisticRegression()

    run(clf, params)

def boosted_tree():
    params = {'max_depth': [5, 6, 7],
                  'n_estimators': [15, 18, 20, 23],
                  #'clf__silent': [False],
                  #'clf__learning_rate': [1e-1, 1e-2, 1e-3],
                  'booster': ['gbtree', 'gblinear'],
                  'objective': ['reg:logistic'] }
    clf = xgb.XGBClassifier()

    run(clf, params)

def random_forest():
    params = {
        'n_estimators': [8, 9],
        'criterion': ['gini', 'entropy'],
        'max_features': [4, 5],
        'max_depth': [7, 8]
    }
    clf = RandomForestClassifier()

    run(clf, params)


def svm_clf():
    params = {  'C': [1e-3, 1e-2, 1e-1, 1, 1e1],
                'gamma': [1e-2, 1e-1, 1],
                'kernel': ['rbf', 'linear', 'sigmoid', 'poly']  }
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
