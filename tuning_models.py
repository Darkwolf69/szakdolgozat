# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:56:47 2024

@author: adamWolf

model_evaluation_utils code is available at:
https://github.com/dipanjanS/text-analytics-with-python/blob/master/New-Second-Edition/Ch05%20-%20Text%20Classification/model_evaluation_utils.py

"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import model_evaluation_utils as meu


def tuning_MultinomialNB():
    # reading prepared train and test corpus
    with open('datasets_list.npy', 'rb') as f:
        train_corpus = np.load(f, allow_pickle=True)
        test_corpus = np.load(f, allow_pickle=True)
        train_label_nums = np.load(f, allow_pickle=True)
        test_label_nums = np.load(f, allow_pickle=True)
        train_label_names = np.load(f, allow_pickle=True)
        test_label_names = np.load(f, allow_pickle=True)


    labels_map = {
        0: 'muggle_world',
        1: 'magic_outside_Hogwarts',
        2: 'Voldemort_story_arch',
        3: 'Hogwarts_classroom_quidditch',
    }
# =============================================================================
# # Tuning the Multinomial Naïve Bayes model
# =============================================================================

    mnb_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                             ('mnb', MultinomialNB())
                             ])
    
    param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
    'mnb__alpha': [1e-5, 1e-4, 1e-2, 1e-1, 1]
    }
    
    gs_mnb = GridSearchCV(mnb_pipeline, param_grid, cv=5, verbose=2)
    gs_mnb = gs_mnb.fit(train_corpus, train_label_names)
    
    
    gs_mnb.best_estimator_.get_params()
    
    {'memory': None,
     'steps': [('tfidf',
                TfidfVectorizer(analyzer='word', max_df=1.0, min_df=1, ngram_range=(1, 2),
                                norm='l2', use_idf=True),
    ('mnb', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)))],
    'tfidf': TfidfVectorizer(analyzer='word', max_df=1.0, min_df=1, ngram_range=(1, 2),
        norm='l2', use_idf=True),
    'mnb': MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True),
    'tfidf__analyzer': 'word', 'tfidf__binary': False, 'tfidf__decode_error':
    'strict',
    'tfidf__dtype': np.float64, 'tfidf__encoding': 'utf-8', 'tfidf__input':
    'content',
    'tfidf__lowercase': True, 'tfidf__max_df': 1.0, 'tfidf__max_features': None,
    'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2), 'tfidf__norm': 'l2',
    'tfidf__preprocessor': None, 'tfidf__smooth_idf': True, 'tfidf__stop_words': None,
    'tfidf__strip_accents': None, 'tfidf__sublinear_tf': False,
    'tfidf__token_pattern': '(?u)\\b\\w\\w+\\b', 'tfidf__tokenizer': None,
    'tfidf__use_idf': True,
    'tfidf__vocabulary': None, 'mnb__alpha': 0.01, 'mnb__class_prior': None,
    'mnb__fit_prior': True}
        
    
    cv_results = gs_mnb.cv_results_
    results_df = pd.DataFrame({'rank': cv_results['rank_test_score'],
                               'params': cv_results['params'],
                               'cv score (mean)': cv_results['mean_test_score'],
                               'cv score (std)': cv_results['std_test_score']}
                              )
    results_df = results_df.sort_values(by=['rank'], ascending=True)
    pd.set_option('display.max_colwidth', 100)
    print(results_df.to_string())
    
    best_mnb_test_score = gs_mnb.score(test_corpus, test_label_names)
    print('Tuned MultinomialNB model Test Accuracy :', best_mnb_test_score)
    print('\n')

# =============================================================================
# Tuning the Logistic Regression model
# =============================================================================
    
    lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(penalty='l2', max_iter=100, random_state=42))
    ])
    
    param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
    'lr__C': [1, 5, 10]
    }
    gs_lr = GridSearchCV(lr_pipeline, param_grid, cv=5, verbose=2)
    gs_lr = gs_lr.fit(train_corpus, train_label_names)
    
    # evaluate best tuned model on the test dataset
    best_lr_test_score = gs_lr.score(test_corpus, test_label_names)
    print('Tuned Logistic Regression model Test Accuracy :', best_lr_test_score)
    print('\n')

# =============================================================================
# Tuning the Linear SVM model
# =============================================================================
    
    svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC(random_state=42, dual='auto'))
    ])
    param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svm__C': [0.01, 0.1, 1, 5]
    }
    
    gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, verbose=2)
    gs_svm = gs_svm.fit(train_corpus, train_label_names)
    # evaluating best tuned model on the test dataset
    best_svm_test_score = gs_svm.score(test_corpus, test_label_names)
    print('Tuned Linear SVM model Test Accuracy :', best_svm_test_score)
    print('\n')
    
# =============================================================================
#   model performance evaluation  
# =============================================================================
    
    #  get accuracy, precision, recall, F1 score
    mnb_predictions = gs_mnb.predict(test_corpus)
    unique_classes = list(set(test_label_names))
    meu.get_metrics(true_labels=test_label_names, predicted_labels=mnb_predictions)
    print('\n')
    print('mnb_predictions: ', mnb_predictions)
    print('unique_classes: ', unique_classes)
    print('\n')
    
    meu.display_classification_report(true_labels=test_label_names,
                                      predicted_labels=mnb_predictions, classes=unique_classes)
    
    # print mapping between class label names and numbers
    label_data_map = {v:k for k, v in labels_map.items()}
    label_map_df = pd.DataFrame(list(label_data_map.items()),
    columns=['Label Name', 'Label Number'])
    print(label_map_df)
    
    # print confusion matrix
    unique_class_nums = label_map_df['Label Number'].values
    mnb_prediction_class_nums = [label_data_map[item] for item in mnb_predictions]
    meu.display_confusion_matrix(true_labels=test_label_nums,
                                        predicted_labels=mnb_prediction_class_nums,
                                        classes=unique_class_nums)
    
    
    
tuning_MultinomialNB()