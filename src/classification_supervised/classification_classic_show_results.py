# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:04:03 2024

@author: adamWolf
"""

from IPython.display import display

import pandas as pd


def show_classification_results(results):
    """
    Display classification results in a Pandas DataFrame.

    Parameters:
        results (dict): A dictionary containing classification results.

    Displays a DataFrame with the following columns:
        Model: Name of the classification model.
        CV Score (TF): Mean cross-validation accuracy using Bag of Words (BOW).
        Test Score (TF): Test accuracy using Bag of Words (BOW).
        CV Score (TF-IDF): Mean cross-validation accuracy using TF-IDF.
        Test Score (TF-IDF): Test accuracy using TF-IDF.
    """
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
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 0)
    pd.set_option('display.precision', 6)
    
    display(results_frame)
