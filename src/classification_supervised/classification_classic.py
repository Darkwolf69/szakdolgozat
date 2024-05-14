# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:30:49 2024

@author: adamWolf
"""

import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def classification_classic(train_corpus, test_corpus, train_label_nums,
                           test_label_nums, train_label_names, test_label_names):
    """
    Perform classification using Multinomial Naive Bayes, Logistic Regression,
    Support Vector Machines, SVM-SGD, Random Forest, and Gradient Boosting Machines
    on datasets extracted with Bag of Words (BOW) and TF-IDF models.

    Parameters:
        train_corpus (numpy.ndarray): Training data corpus.
        test_corpus (numpy.ndarray): Test data corpus.
        train_label_nums (numpy.ndarray): Numeric labels for training data.
        test_label_nums (numpy.ndarray): Numeric labels for test data.
        train_label_names (numpy.ndarray): Categorical labels for training data.
        test_label_names (numpy.ndarray): Categorical labels for test data.

    Returns:
        results (dict): A dictionary containing the following classification results:
            mnb_bow_cv_mean_score (float): Mean cross-validation accuracy of Naïve Bayes with BOW.
            mnb_bow_test_score (float): Test accuracy of Naïve Bayes with BOW.
            mnb_tfidf_cv_mean_score (float): Mean cross-validation accuracy of Naïve Bayes with TF-IDF.
            mnb_tfidf_test_score (float): Test accuracy of Naïve Bayes with TF-IDF.
            lr_bow_cv_mean_score (float): Mean cross-validation accuracy of Logistic Regression with BOW.
            lr_bow_test_score (float): Test accuracy of Logistic Regression with BOW.
            lr_tfidf_cv_mean_score (float): Mean cross-validation accuracy of Logistic Regression with TF-IDF.
            lr_tfidf_test_score (float): Test accuracy of Logistic Regression with TF-IDF.
            svm_bow_cv_mean_score (float): Mean cross-validation accuracy of SVM with BOW.
            svm_bow_test_score (float): Test accuracy of SVM with BOW.
            svm_tfidf_cv_mean_score (float): Mean cross-validation accuracy of SVM with TF-IDF.
            svm_tfidf_test_score (float): Test accuracy of SVM with TF-IDF.
            svmsgd_bow_cv_mean_score (float): Mean cross-validation accuracy of SVM with SGD and BOW.
            svmsgd_bow_test_score (float): Test accuracy of SVM with SGD and BOW.
            svmsgd_tfidf_cv_mean_score (float): Mean cross-validation accuracy of SVM with SGD and TF-IDF.
            svmsgd_tfidf_test_score (float): Test accuracy of SVM with SGD and TF-IDF.
            rfc_bow_cv_mean_score (float): Mean cross-validation accuracy of Random Forest with BOW.
            rfc_bow_test_score (float): Test accuracy of Random Forest with BOW.
            rfc_tfidf_cv_mean_score (float): Mean cross-validation accuracy of Random Forest with TF-IDF.
            rfc_tfidf_test_score (float): Test accuracy of Random Forest with TF-IDF.
            gbc_bow_cv_mean_score (float): Mean cross-validation accuracy of Gradient Boosting with BOW.
            gbc_bow_test_score (float): Test accuracy of Gradient Boosting with BOW.
            gbc_tfidf_cv_mean_score (float): Mean cross-validation accuracy of Gradient Boosting with TF-IDF.
            gbc_tfidf_test_score (float): Test accuracy of Gradient Boosting with TF-IDF.
        writes the results dict in a file.
    """
        
    if not all(isinstance(i, np.ndarray) for i in [train_corpus, test_corpus, train_label_nums,
                                                      test_label_nums, train_label_names, test_label_names]):
        raise AttributeError('Invalid input')
    
# =============================================================================
# Bag of Words model
# =============================================================================

    # build BOW features on train articles
    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
    cv_train_features = cv.fit_transform(train_corpus)
    # transform test articles into features
    cv_test_features = cv.transform(test_corpus)
    
    print('\n')
    print('=============================================================================')
    print('Classification with Bag of Words model')
    print('=============================================================================')
    print('\n')
    
    # Naïve Bayes Classifier
    mnb = MultinomialNB(alpha=1)
    mnb.fit(cv_train_features, train_label_names)
    mnb_bow_cv_scores = cross_val_score(mnb, cv_train_features, train_label_names, cv=5)
    mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)
    print('Naïve Bayes Classifier')
    print(f'CV Accuracy (5-fold): {mnb_bow_cv_scores}')
    print(f'Mean CV Accuracy: {mnb_bow_cv_mean_score}')
    mnb_bow_test_score = mnb.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {mnb_bow_test_score}')
    print('\n')
    
    # Logistic Regression
    lr = LogisticRegression(penalty='l2', max_iter=1100, C=1, random_state=42)
    lr.fit(cv_train_features, train_label_names)
    lr_bow_cv_scores = cross_val_score(lr, cv_train_features, train_label_names, cv=5)
    lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
    print('Logistic Regression')
    print(f'CV Accuracy (5-fold): {lr_bow_cv_scores}')
    print(f'Mean CV Accuracy: {lr_bow_cv_mean_score}')
    lr_bow_test_score = lr.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {lr_bow_test_score}')
    print('\n')
    
    # Support Vector Machines
    svm = LinearSVC(penalty='l2', max_iter=300, C=1, random_state=42, dual='auto')
    svm.fit(cv_train_features, train_label_names)
    svm_bow_cv_scores = cross_val_score(svm, cv_train_features, train_label_names, cv=5)
    svm_bow_cv_mean_score = np.mean(svm_bow_cv_scores)
    print('Support Vector Machines')
    print(f'CV Accuracy (5-fold): {svm_bow_cv_scores}')
    print(f'Mean CV Accuracy: {svm_bow_cv_mean_score}')
    svm_bow_test_score = svm.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {svm_bow_test_score}')
    print('\n')
    
    # SVM with Stochastic Gradient Descent
    svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=200, random_state=42)
    svm_sgd.fit(cv_train_features, train_label_names)
    svmsgd_bow_cv_scores = cross_val_score(svm_sgd, cv_train_features, train_label_names, cv=5)
    svmsgd_bow_cv_mean_score = np.mean(svmsgd_bow_cv_scores)
    print('SVM with Stochastic Gradient Descent')
    print(f'CV Accuracy (5-fold): {svmsgd_bow_cv_scores}')
    print(f'Mean CV Accuracy: {svmsgd_bow_cv_mean_score}')
    svmsgd_bow_test_score = svm_sgd.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {svmsgd_bow_test_score}')
    print('\n')
    
    # Random Forest
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(cv_train_features, train_label_names)
    rfc_bow_cv_scores = cross_val_score(rfc, cv_train_features, train_label_names, cv=5)
    rfc_bow_cv_mean_score = np.mean(rfc_bow_cv_scores)
    print('Random Forest')
    print(f'CV Accuracy (5-fold): {rfc_bow_cv_scores}')
    print(f'Mean CV Accuracy: {rfc_bow_cv_mean_score}')
    rfc_bow_test_score = rfc.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {rfc_bow_test_score}')
    print('\n')
    
    # Gradient Boosting Machines
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(cv_train_features, train_label_names)
    gbc_bow_cv_scores = cross_val_score(gbc, cv_train_features, train_label_names, cv=5)
    gbc_bow_cv_mean_score = np.mean(gbc_bow_cv_scores)
    print('Gradient Boosting Machines')
    print(f'CV Accuracy (5-fold): {gbc_bow_cv_scores}')
    print(f'Mean CV Accuracy: {gbc_bow_cv_mean_score}')
    gbc_bow_test_score = gbc.score(cv_test_features, test_label_names)
    print(f'Test Accuracy: {gbc_bow_test_score}')
    
    
# =============================================================================
# TF-IDF model    
# =============================================================================
    
    # build TF-IDF features on train articles
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
    tv_train_features = tv.fit_transform(train_corpus)
    # transform test articles into features
    tv_test_features = tv.transform(test_corpus)
    
    print('\n')
    print('=============================================================================')
    print('Classification with TF-IDF model')
    print('=============================================================================')
    print('\n')
    
    # Naïve Bayes
    mnb = MultinomialNB(alpha=1)
    mnb.fit(tv_train_features, train_label_names)
    mnb_tfidf_cv_scores = cross_val_score(mnb, tv_train_features, train_label_names, cv=5)
    mnb_tfidf_cv_mean_score = np.mean(mnb_tfidf_cv_scores)
    print('Naïve Bayes Classifier')
    print(f'CV Accuracy (5-fold): {mnb_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {mnb_tfidf_cv_mean_score}')
    mnb_tfidf_test_score = mnb.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {mnb_tfidf_test_score}')
    print('\n')
    
    # Logistic Regression
    lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
    lr.fit(tv_train_features, train_label_names)
    lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, train_label_names, cv=5)
    lr_tfidf_cv_mean_score = np.mean(lr_tfidf_cv_scores)
    print('Logistic Regression')
    print(f'CV Accuracy (5-fold): {lr_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {lr_tfidf_cv_mean_score}')
    lr_tfidf_test_score = lr.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {lr_tfidf_test_score}')
    print('\n')
    
    # Support Vector Machines
    svm = LinearSVC(penalty='l2', C=1, random_state=42, dual='auto')
    svm.fit(tv_train_features, train_label_names)
    svm_tfidf_cv_scores = cross_val_score(svm, tv_train_features, train_label_names, cv=5)
    svm_tfidf_cv_mean_score = np.mean(svm_tfidf_cv_scores)
    print('Support Vector Machines')
    print(f'CV Accuracy (5-fold): {svm_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {svm_tfidf_cv_mean_score}')
    svm_tfidf_test_score = svm.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {svm_tfidf_test_score}')
    print('\n')
    
    # SVM with Stochastic Gradient Descent
    svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
    svm_sgd.fit(tv_train_features, train_label_names)
    svmsgd_tfidf_cv_scores = cross_val_score(svm_sgd, tv_train_features, train_label_names, cv=5)
    svmsgd_tfidf_cv_mean_score = np.mean(svmsgd_tfidf_cv_scores)
    print('SVM with Stochastic Gradient Descent')
    print(f'CV Accuracy (5-fold): {svmsgd_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {svmsgd_tfidf_cv_mean_score}')
    svmsgd_tfidf_test_score = svm_sgd.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {svmsgd_tfidf_test_score}')
    print('\n')
    
    # Random Forest
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(tv_train_features, train_label_names)
    rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, train_label_names, cv=5)
    rfc_tfidf_cv_mean_score = np.mean(rfc_tfidf_cv_scores)
    print('Random Forest')
    print(f'CV Accuracy (5-fold): {rfc_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {rfc_tfidf_cv_mean_score}')
    rfc_tfidf_test_score = rfc.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {rfc_tfidf_test_score}')
    print('\n')
        
    # Gradient Boosting
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(tv_train_features, train_label_names)
    gbc_tfidf_cv_scores = cross_val_score(gbc, tv_train_features, train_label_names, cv=5)
    gbc_tfidf_cv_mean_score = np.mean(gbc_tfidf_cv_scores)
    print('Gradient Boosting Machines')
    print(f'CV Accuracy (5-fold): {gbc_tfidf_cv_scores}')
    print(f'Mean CV Accuracy: {gbc_tfidf_cv_mean_score}')
    gbc_tfidf_test_score = gbc.score(tv_test_features, test_label_names)
    print(f'Test Accuracy: {gbc_tfidf_test_score}')


    results = {
                'mnb_bow_cv_mean_score': mnb_bow_cv_mean_score,
                'mnb_bow_test_score': mnb_bow_test_score,
                'mnb_tfidf_cv_mean_score': mnb_tfidf_cv_mean_score,
                'mnb_tfidf_test_score': mnb_tfidf_test_score,
                'lr_bow_cv_mean_score': lr_bow_cv_mean_score,
                'lr_bow_test_score': lr_bow_test_score,
                'lr_tfidf_cv_mean_score': lr_tfidf_cv_mean_score,
                'lr_tfidf_test_score': lr_tfidf_test_score,
                'svm_bow_cv_mean_score': svm_bow_cv_mean_score,
                'svm_bow_test_score': svm_bow_test_score,
                'svm_tfidf_cv_mean_score': svm_tfidf_cv_mean_score,
                'svm_tfidf_test_score': svm_tfidf_test_score,
                'svmsgd_bow_cv_mean_score': svmsgd_bow_cv_mean_score,
                'svmsgd_bow_test_score': svmsgd_bow_test_score,
                'svmsgd_tfidf_cv_mean_score': svmsgd_tfidf_cv_mean_score,
                'svmsgd_tfidf_test_score': svmsgd_tfidf_test_score,
                'rfc_bow_cv_mean_score': rfc_bow_cv_mean_score,
                'rfc_bow_test_score': rfc_bow_test_score,
                'rfc_tfidf_cv_mean_score': rfc_tfidf_cv_mean_score,
                'rfc_tfidf_test_score': rfc_tfidf_test_score,
                'gbc_bow_cv_mean_score': gbc_bow_cv_mean_score,
                'gbc_bow_test_score': gbc_bow_test_score,
                'gbc_tfidf_cv_mean_score': gbc_tfidf_cv_mean_score,
                'gbc_tfidf_test_score': gbc_tfidf_test_score,
    
        }
    
    
    with open('classification_classic_results.txt', 'w') as file: 
        file.write(json.dumps(results))
