from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_validate, RandomizedSearchCV
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier 
from bs4 import BeautifulSoup
from random import shuffle
from sklearn.decomposition import NMF, LatentDirichletAllocation

import xgboost as xgb
import numpy as np
import lightgbm as lgb 

import csv
import time

np.warnings.filterwarnings('ignore')

# load dataset and eliminate html tags
def load_data(filename):
    #filename = 'train_and_test.csv'
    #filename = 'data/train.csv'

    data = []
    valid_ = []
    arg_valid = []
    complaint_misrep = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row['complain'])
            valid_.append(row['is complaint valid'])
            arg_valid.append(row['is argumentation valid?'])
            complaint_misrep.append(row['is complaint misrepresenting?'])
        
    # print(data[1])
    complaint_list = [BeautifulSoup(item, "lxml").text for item in data]
    complaint_fe = filter_empty(complaint_list, valid_)
    complaint = list(filter(None, complaint_fe))

    valid = list(filter(None, valid_))
    
    combo = list(zip(complaint, valid))
    shuffle(combo)
    complaint[:], valid[:] = zip(*combo)

    return complaint, valid #, arg_valid, complaint_misrep

def eval(targ, pred, labels):
    #print(metrics.classification_report(targ, pred, target_names=labels))
    print("F1: {}".format(metrics.f1_score(targ, pred, pos_label='Y')))


def mapper(idx):
    if idx == 'Y':
        return 1
    elif idx == 'N':
        return 0
    else: return ''

def filter_empty(l1, l2):
    new_l1 = ['' if i != 'Y' and i != 'N' else o for i, o in zip(l2, l1)]
    return new_l1

def svm_clf():
    '''
    complaints_train, validity_train, _, _ = load_data('data/train.csv')
    complaints_test, validity_test, _, _ = load_data('data/test.csv')
    '''
    c_tr, v_tr = load_data('data/train_and_test.csv')
    
    clf = svm.SVC()#kernel='rbf')
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                  'gamma': [1e-2, 1e-1, 1, 1e1, 1e2],
                  'kernel': ['linear', 'rbf', 'sigmoid', 'poly']
                  }
    
    # GridSearch takes obnoxiously long to complete, but results are only slightly nicer than Randomized search
    # However, soem research argues otherwise 
    #gs = GridSearchCV(clf, param_grid)
    gs = RandomizedSearchCV(clf, param_distributions=param_grid,
                            n_iter=18)
    
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=0.05, stop_words='english' )),
                        ('tfidf', TfidfTransformer()),
                        ('clf', gs)])
                        #('clf', clf)])
    
    #pipeline.fit(complaints_train, validity_train)
    #y_pred = pipeline.predict(complaints_test) 

    predicted = cross_val_predict(pipeline, c_tr, v_tr, cv=10)

    return predicted, v_tr

# logistic regression 
def regression():

    c_tr, v_tr = load_data('data/train_and_test.csv')

    param_grid = { 'penalty': ['l1', 'l2'],
                   'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                   'class_weight': ["balanced", ] 
                   }
    #classifiers
    reg = GridSearchCV(linear_model.LogisticRegression(), param_grid)

    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=0.05, stop_words='english' )),
                        ('tfidf', TfidfTransformer()), 
                        ('clf', reg)])  # logistic regression 

    predicted = cross_val_predict(pipeline, c_tr, v_tr, cv=10)
    
    return predicted, v_tr

def boosted_tree():

    comp_tr, val_tr = load_data('data/train_and_test.csv')

    param_list = {'max_depth': [5, 6, 7],
                  'n_estimators': [20],
                  #'silent': [False],
                  #'learning_rate': [1e-1, 1e-2, 1e-3],
                  'booster': ['gbtree', 'gblinear'],
                  'objective': ['reg:logistic']}

    clf = GridSearchCV(xgb.XGBClassifier(), param_list)

    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=0.05, stop_words='english')),
                        ('tfidf', TfidfTransformer()), 
                        ('clf', clf) ] )

    """
        Notes to self:
        -> cross_val_predict has params shuffle=False by default and uses KFold 
    """
    predicted = cross_val_predict(pipeline, comp_tr, val_tr, cv=10)

    #scores = cross_validate(pipeline, x_tr, y_tr, cv=10)
    #print(scores.keys())
    #print(clf.best_estimator_)

    return predicted, val_tr


def lda():

    comp_tr, val_tr = load_data('data/train_and_test.csv')

    counts = CountVectorizer(ngram_range=(1, 3), max_df=0.9, 
                             min_df=0.05, stop_words='english')
    c_fit = counts.fit_transform(comp_tr, y=val_tr)
    feat_names = counts.get_feature_names()


def rand_forest():
    c_tr, v_tr = load_data('data/train_and_test.csv')

    print(len(c_tr), len(v_tr))

    params = {
        'n_estimators': [8, 9, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': [4, 5],
        'max_depth': [8, 9, 10]
    }
    #clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=25)
    clf = GridSearchCV(RandomForestClassifier(), param_grid=params)

    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=0.05, stop_words='english')),
                        ('tfidf', TfidfTransformer()), 
                        ('clf', clf) ] )
    
    predicted = cross_val_predict(pipeline, c_tr, v_tr, cv=10)
    #print(clf.best_estimator_)

    return predicted, v_tr

def main():
    labels = [ 'Y', 'N' ]
    
    st_init = st = time.time()
    #print("Started: {}".format(st))
    
    pred, val = boosted_tree()
    print("\n Boosted Trees:")
    eval(val, pred, labels)
    print("Time: {} seconds\n".format( time.time() - st_init))
    
    st = time.time()

    pred, val = regression()
    print("\n Logistic Regression:")
    eval(val, pred, labels)
    print("Time: {} seconds\n".format( time.time() - st) )
    
    st = time.time()
    y_pred, y_true = rand_forest()
    print("Random Forests:")
    eval(y_pred, y_true, labels)
    print("Time: {} seconds\n".format( time.time() - st ))
    
    st = time.time()
    pred, val = svm_clf()
    print("\n SVM:")
    eval(val, pred, labels)
    print("Time: {} seconds\n".format( time.time() - st))
    
    print("That took {} total seconds\n".format( time.time() - st_init ))


main()
