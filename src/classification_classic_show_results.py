# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:04:03 2024

@author: adamWolf

shows results of text classification using classic (Bag of Words, TF-IDF) feature engineering models

"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from IPython.display import display


def show_classification_results():
    with open('classification_classic_results.txt', 'r') as file:
        results = json.load(file, )
        
    results_frame = pd.DataFrame([['Naive Bayes', results.get('mnb_bow_cv_mean_score'), results.get('mnb_bow_test_score'),
        results.get('mnb_tfidf_cv_mean_score'), results.get('mnb_tfidf_test_score')],
        ['Logistic Regression', results.get('lr_bow_cv_mean_score'), results.get('lr_bow_test_score'),
        results.get('lr_tfidf_cv_mean_score'), results.get('lr_tfidf_test_score')],
        ['Linear SVM', results.get('svm_bow_cv_mean_score'), results.get('svm_bow_test_score'),
        results.get('svm_tfidf_cv_mean_score'), results.get('svm_tfidf_test_score')],
        ['Linear SVM (SGD)', results.get('svmsgd_bow_cv_mean_score'), results.get('svmsgd_bow_test_score'),
        results.get('svmsgd_tfidf_cv_mean_score'), results.get('svmsgd_tfidf_test_score')],
        ['Random Forest', results.get('rfc_bow_cv_mean_score'), results.get('rfc_bow_test_score'),
        results.get('rfc_tfidf_cv_mean_score'), results.get('rfc_tfidf_test_score')],
        ['Gradient Boosted Machines', results.get('gbc_bow_cv_mean_score'), results.get('gbc_bow_test_score'),
        results.get('gbc_tfidf_cv_mean_score'), results.get('gbc_tfidf_test_score')]],
        columns=['Model', 'CV Score (TF)', 'Test Score (TF)',
        'CV Score (TF-IDF)', 'Test Score (TF-IDF)'],
        ).T
    
    # pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 0)
    pd.set_option('display.precision', 6)
    
    # plotting the results
    # fig = plt.figure(figsize = (8, .2))
    # ax = fig.add_subplot(111)
    
    # ax.table(cellText = results_frame.values, rowLabels = results_frame.index, 
    #      colLabels = results_frame.columns, cellLoc='center')
    # ax.set_title('Classification results with BOW and TF-IDF')
    # plt.show()
    
    display(results_frame)

show_classification_results()