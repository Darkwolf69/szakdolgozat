# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:30:49 2024

@author: adamWolf

uding bag of Words, the term frequency-based feature engineering model
model
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

def extract_features_to_classification():
    with open('datasets_list.npy', 'rb') as f:
        train_corpus = np.load(f, allow_pickle=True)
        test_corpus = np.load(f, allow_pickle=True)
        train_label_nums = np.load(f, allow_pickle=True)
        test_label_nums = np.load(f, allow_pickle=True)
        train_label_names = np.load(f, allow_pickle=True)
        test_label_names = np.load(f, allow_pickle=True)
    
    # build BOW features on train articles
    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
    cv_train_features = cv.fit_transform(train_corpus)
    # transform test articles into features
    cv_test_features = cv.transform(test_corpus)
    print('BOW model:> Train features shape:', cv_train_features.shape,
          ' Test features shape:', cv_test_features.shape)
    
    
    # Naïve Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB(alpha=1)
    mnb.fit(cv_train_features, train_label_names)
    mnb_bow_cv_scores = cross_val_score(mnb, cv_train_features, train_label_names, cv=5)
    mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)
    print('Naïve Bayes Classifier\n')
    print('CV Accuracy (5-fold):', mnb_bow_cv_scores)
    print('Mean CV Accuracy:', mnb_bow_cv_mean_score)
    mnb_bow_test_score = mnb.score(cv_test_features, test_label_names)
    print('Test Accuracy:', mnb_bow_test_score)
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
    lr.fit(cv_train_features, train_label_names)
    lr_bow_cv_scores = cross_val_score(lr, cv_train_features, train_label_names, cv=5)
    lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
    print('Logistic Regression\n')
    print('CV Accuracy (5-fold):', lr_bow_cv_scores)
    print('Mean CV Accuracy:', lr_bow_cv_mean_score)
    lr_bow_test_score = lr.score(cv_test_features, test_label_names)
    print('Test Accuracy:', lr_bow_test_score)
    
    # Support Vector Machines
    from sklearn.svm import LinearSVC
    svm = LinearSVC(penalty='l2', C=1, random_state=42)
    svm.fit(cv_train_features, train_label_names)
    svm_bow_cv_scores = cross_val_score(svm, cv_train_features, train_label_names, cv=5)
    svm_bow_cv_mean_score = np.mean(svm_bow_cv_scores)
    print('Support Vector Machines\n')
    print('CV Accuracy (5-fold):', svm_bow_cv_scores)
    print('Mean CV Accuracy:', svm_bow_cv_mean_score)
    svm_bow_test_score = svm.score(cv_test_features, test_label_names)
    print('Test Accuracy:', svm_bow_test_score)
    
    # SVM with Stochastic Gradient Descent
    from sklearn.linear_model import SGDClassifier
    svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42)
    svm_sgd.fit(cv_train_features, train_label_names)
    svmsgd_bow_cv_scores = cross_val_score(svm_sgd, cv_train_features, train_label_names, cv=5)
    svmsgd_bow_cv_mean_score = np.mean(svmsgd_bow_cv_scores)
    print('SVM with Stochastic Gradient Descent\n')
    print('CV Accuracy (5-fold):', svmsgd_bow_cv_scores)
    print('Mean CV Accuracy:', svmsgd_bow_cv_mean_score)
    svmsgd_bow_test_score = svm_sgd.score(cv_test_features, test_label_names)
    print('Test Accuracy:', svmsgd_bow_test_score)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(cv_train_features, train_label_names)
    rfc_bow_cv_scores = cross_val_score(rfc, cv_train_features, train_label_names, cv=5)
    rfc_bow_cv_mean_score = np.mean(rfc_bow_cv_scores)
    print('Random Forest\n')
    print('CV Accuracy (5-fold):', rfc_bow_cv_scores)
    print('Mean CV Accuracy:', rfc_bow_cv_mean_score)
    rfc_bow_test_score = rfc.score(cv_test_features, test_label_names)
    print('Test Accuracy:', rfc_bow_test_score)
    
    # Gradient Boosting Machines
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(cv_train_features, train_label_names)
    gbc_bow_cv_scores = cross_val_score(gbc, cv_train_features, train_label_names, cv=5)
    gbc_bow_cv_mean_score = np.mean(gbc_bow_cv_scores)
    print('Gradient Boosting Machines\n')
    print('CV Accuracy (5-fold):', gbc_bow_cv_scores)
    print('Mean CV Accuracy:', gbc_bow_cv_mean_score)
    gbc_bow_test_score = gbc.score(cv_test_features, test_label_names)
    print('Test Accuracy:', gbc_bow_test_score)
    
    
    

extract_features_to_classification()