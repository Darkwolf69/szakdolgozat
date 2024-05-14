# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 02:40:49 2024

@author: adamWolf
"""

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def classification_advanced(train_corpus, test_corpus, train_label_nums,
                            test_label_nums, train_label_names, test_label_names,
                            avg_w2v_train_features, avg_w2v_test_features,
                            train_glove_features, test_glove_features,
                            avg_ft_train_features, avg_ft_test_features):
    """
    Perform classification using Gaussian Naive Bayes, Logistic Regression,
    Support Vector Machines, SVM-SGD, Random Forest, and Gradient Boosting Machines on
    Word2Vec, GloVe, and FastText datasets. It also includes a neural network model using Multi-layer
    Perceptron (MLP) with two hidden layers on each dataset.

    Parameters:
        train_corpus (np.ndarray): Array containing training data.
        test_corpus (np.ndarray): Array containing testing data.
        train_label_nums (np.ndarray): Array containing numerical labels for training data.
        test_label_nums (np.ndarray): Array containing numerical labels for testing data.
        train_label_names (np.ndarray): Array containing label names for training data.
        test_label_names (np.ndarray): Array containing label names for testing data.
        avg_w2v_train_features (np.ndarray): Word2Vec features for training data.
        avg_w2v_test_features (np.ndarray): Word2Vec features for testing data.
        train_glove_features (np.ndarray): GloVe features for training data.
        test_glove_features (np.ndarray): GloVe features for testing data.
        avg_ft_train_features (np.ndarray): FastText features for training data.
        avg_ft_test_features (np.ndarray): FastText features for testing data.

    Returns:
        None (writes results of the classifications on the console)

    Raises:
        AttributeError: If any of the input is not a numpy array.
    """
    
    if not all(isinstance(i, np.ndarray) for i in [train_corpus, test_corpus, train_label_nums,
                                                      test_label_nums, train_label_names, test_label_names,
                                                      avg_w2v_train_features, avg_w2v_test_features,
                                                      train_glove_features, test_glove_features,
                                                      avg_ft_train_features, avg_ft_test_features]):
        raise AttributeError('Invalid input')
        
# =============================================================================
#  Classification with Word2Vec dataset
# =============================================================================

    # Gaussian Naïve Bayes Classifier with Word2Vec dataset
    clf = GaussianNB()
    clf.fit(avg_w2v_train_features, train_label_names)
    clf_w2v_cv_scores = cross_val_score(clf, avg_w2v_train_features, train_label_names, cv=5)
    clf_w2v_cv_mean_score = np.mean(clf_w2v_cv_scores)
    print('Gaussian Naïve Bayes Classifier with Word2Vec dataset')
    print(f'CV Accuracy (5-fold): {clf_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {clf_w2v_cv_mean_score}')
    clf_w2v_test_score = clf.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {clf_w2v_test_score}')
    print('\n')
    
    # Logistic Regression with Word2Vec dataset
    lr = LogisticRegression(penalty='l2', max_iter=1100, C=1, random_state=42)
    lr.fit(avg_w2v_train_features, train_label_names)
    lr_w2v_cv_scores = cross_val_score(lr, avg_w2v_train_features, train_label_names, cv=5)
    lr_w2v_cv_mean_score = np.mean(lr_w2v_cv_scores)
    print('Logistic Regression with Word2Vec dataset')
    print(f'CV Accuracy (5-fold): {lr_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {lr_w2v_cv_mean_score}')
    lr_w2v_test_score = lr.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {lr_w2v_test_score}')
    print('\n')
    
    # Support Vector Machines with Word2Vec dataset
    svm = LinearSVC(penalty='l2', max_iter=300, C=1, random_state=42, dual='auto')
    svm.fit(avg_w2v_train_features, train_label_names)
    svm_w2v_cv_scores = cross_val_score(svm, avg_w2v_train_features, train_label_names, cv=5)
    svm_w2v_cv_mean_score = np.mean(svm_w2v_cv_scores)
    print('Support Vector Machines with Word2Vec dataset')
    print(f'CV Accuracy (5-fold): {svm_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {svm_w2v_cv_mean_score}')
    svm_w2v_test_score = svm.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {svm_w2v_test_score}')
    print('\n')
    
    # SVM-SGD model with Word2Vec dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(avg_w2v_train_features, train_label_names)
    svm_w2v_cv_scores = cross_val_score(svm, avg_w2v_train_features, train_label_names, cv=5)
    svm_w2v_cv_mean_score = np.mean(svm_w2v_cv_scores)
    print('\n')
    print('SVM-SGD model with Word2Vec dataset')
    print(f'CV Accuracy (5-fold): {svm_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {svm_w2v_cv_mean_score}')
    svm_w2v_test_score = svm.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {svm_w2v_test_score}')
    print('\n')
    
    # Random Forest with Word2Vec dataset
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(avg_w2v_train_features, train_label_names)
    rfc_w2v_cv_scores = cross_val_score(rfc, avg_w2v_train_features, train_label_names, cv=5)
    rfc_w2v_cv_mean_score = np.mean(rfc_w2v_cv_scores)
    print('Random Forest with Word2Vec dataset')
    print(f'CV Accuracy (5-fold): {rfc_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {rfc_w2v_cv_mean_score}')
    rfc_w2v_test_score = rfc.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {rfc_w2v_test_score}')
    print('\n')
    
    # Gradient Boosting Machines with Word2Vec dataset
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(avg_w2v_train_features, train_label_names)
    gbc_w2v_cv_scores = cross_val_score(gbc, avg_w2v_train_features, train_label_names, cv=5)
    gbc_w2v_cv_mean_score = np.mean(gbc_w2v_cv_scores)
    print('Gradient Boosting Machines with Word2Vec dataset}')
    print(f'CV Accuracy (5-fold): {gbc_w2v_cv_scores}')
    print(f'Mean CV Accuracy: {gbc_w2v_cv_mean_score}')
    gbc_w2v_test_score = gbc.score(avg_w2v_test_features, test_label_names)
    print(f'Test Accuracy: {gbc_w2v_test_score}')
    print('\n')
    
    
# =============================================================================
#  Classification with GloVe dataset
# =============================================================================
    
    # Gaussian Naïve Bayes Classifier with GloVe dataset
    clf = GaussianNB()
    clf.fit(train_glove_features, train_label_names)
    clf_glove_cv_scores = cross_val_score(clf, train_glove_features, train_label_names, cv=5)
    clf_glove_cv_mean_score = np.mean(clf_glove_cv_scores)
    print('Gaussian Naïve Bayes Classifier with GloVe dataset')
    print(f'CV Accuracy (5-fold): {clf_glove_cv_scores}')
    print(f'Mean CV Accuracy: {clf_glove_cv_mean_score}')
    clf_glove_test_score = clf.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {clf_glove_test_score}')
    print('\n')
    
    # Logistic Regression with GloVe dataset
    lr = LogisticRegression(penalty='l2', max_iter=1100, C=1, random_state=42)
    lr.fit(train_glove_features, train_label_names)
    lr_glove_cv_scores = cross_val_score(lr, train_glove_features, train_label_names, cv=5)
    lr_glove_cv_mean_score = np.mean(lr_glove_cv_scores)
    print('Logistic Regression with GloVe dataset')
    print(f'CV Accuracy (5-fold): {lr_glove_cv_scores}')
    print(f'Mean CV Accuracy: {lr_glove_cv_mean_score}')
    lr_glove_test_score = lr.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {lr_glove_test_score}')
    print('\n')
    
    # Support Vector Machines with GloVe dataset
    svm = LinearSVC(penalty='l2', max_iter=300, C=1, random_state=42, dual='auto')
    svm.fit(train_glove_features, train_label_names)
    svm_glove_cv_scores = cross_val_score(svm, train_glove_features, train_label_names, cv=5)
    svm_glove_cv_mean_score = np.mean(svm_glove_cv_scores)
    print('Support Vector Machines with GloVe dataset')
    print(f'CV Accuracy (5-fold): {svm_glove_cv_scores}')
    print(f'Mean CV Accuracy: {svm_glove_cv_mean_score}')
    svm_glove_test_score = svm.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {svm_glove_test_score}')
    print('\n')

    # SVM-SGD model with GloVe dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(train_glove_features, train_label_names)
    svm_glove_cv_scores = cross_val_score(svm, train_glove_features, train_label_names, cv=5)
    svm_glove_cv_mean_score = np.mean(svm_glove_cv_scores)
    print('SVM-SGD model with GloVe dataset')
    print(f'CV Accuracy (5-fold): {svm_glove_cv_scores}')
    print(f'Mean CV Accuracy: {svm_glove_cv_mean_score}')
    svm_glove_test_score = svm.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {svm_glove_test_score}')
    print('\n')

    # Random Forest with GloVe dataset
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(train_glove_features, train_label_names)
    rfc_glove_cv_scores = cross_val_score(rfc, train_glove_features, train_label_names, cv=5)
    rfc_glove_cv_mean_score = np.mean(rfc_glove_cv_scores)
    print('Random Forest with GloVe dataset')
    print(f'CV Accuracy (5-fold): {rfc_glove_cv_scores}')
    print(f'Mean CV Accuracy: {rfc_glove_cv_mean_score}')
    rfc_glove_test_score = rfc.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {rfc_glove_test_score}')
    print('\n')

    # Gradient Boosting Machines with GloVe dataset
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(train_glove_features, train_label_names)
    gbc_glove_cv_scores = cross_val_score(gbc, train_glove_features, train_label_names, cv=5)
    gbc_glove_cv_mean_score = np.mean(gbc_glove_cv_scores)
    print('Gradient Boosting Machines with GloVe dataset')
    print(f'CV Accuracy (5-fold): {gbc_glove_cv_scores}')
    print(f'Mean CV Accuracy: {gbc_glove_cv_mean_score}')
    gbc_glove_test_score = gbc.score(test_glove_features, test_label_names)
    print(f'Test Accuracy: {gbc_glove_test_score}')
    print('\n')
    
    
# =============================================================================
#  Classification with FastText dataset
# =============================================================================

    # Gaussian Naïve Bayes Classifier with FastText dataset
    clf = GaussianNB()
    clf.fit(avg_ft_train_features, train_label_names)
    clf_ft_cv_scores = cross_val_score(clf, avg_ft_train_features, train_label_names, cv=5)
    clf_ft_cv_mean_score = np.mean(clf_ft_cv_scores)
    print('Naïve Bayes Classifier with FastText dataset')
    print(f'CV Accuracy (5-fold): {clf_ft_cv_scores}')
    print(f'Mean CV Accuracy: {clf_ft_cv_mean_score}')
    clf_ft_test_score = clf.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {clf_ft_test_score}')
    print('\n')

    # Logistic Regression with FastText dataset
    lr = LogisticRegression(penalty='l2', max_iter=1100, C=1, random_state=42)
    lr.fit(avg_ft_train_features, train_label_names)
    lr_ft_cv_scores = cross_val_score(lr, avg_ft_train_features, train_label_names, cv=5)
    lr_ft_cv_mean_score = np.mean(lr_ft_cv_scores)
    print('Logistic Regression with FastText dataset')
    print(f'CV Accuracy (5-fold): {lr_ft_cv_scores}')
    print(f'Mean CV Accuracy: {lr_ft_cv_mean_score}')
    lr_ft_test_score = lr.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {lr_ft_test_score}')
    print('\n')

    # Support Vector Machines with FastText dataset
    svm = LinearSVC(penalty='l2', max_iter=300, C=1, random_state=42, dual='auto')
    svm.fit(avg_ft_train_features, train_label_names)
    svm_ft_cv_scores = cross_val_score(svm, avg_ft_train_features, train_label_names, cv=5)
    svm_ft_cv_mean_score = np.mean(svm_ft_cv_scores)
    print('Support Vector Machines with FastText dataset')
    print(f'CV Accuracy (5-fold): {svm_ft_cv_scores}')
    print(f'Mean CV Accuracy: {svm_ft_cv_mean_score}')
    svm_ft_test_score = svm.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {svm_ft_test_score}')
    print('\n')

    # SVM-SGD model with FastText dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(avg_ft_train_features, train_label_names)
    svm_ft_cv_scores = cross_val_score(svm, avg_ft_train_features, train_label_names, cv=5)
    svm_ft_cv_mean_score = np.mean(svm_ft_cv_scores)
    print('SVM-SGD model with FastText dataset')
    print(f'CV Accuracy (5-fold): {svm_ft_cv_scores}')
    print(f'Mean CV Accuracy: {svm_ft_cv_mean_score}')
    svm_ft_test_score = svm.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {svm_ft_test_score}')
    print('\n')

    # Random Forest with FastText dataset
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(avg_ft_train_features, train_label_names)
    rfc_ft_cv_scores = cross_val_score(rfc, avg_ft_train_features, train_label_names, cv=5)
    rfc_ft_cv_mean_score = np.mean(rfc_ft_cv_scores)
    print('Random Forest with FastText dataset')
    print(f'CV Accuracy (5-fold): {rfc_ft_cv_scores}')
    print(f'Mean CV Accuracy: {rfc_ft_cv_mean_score}')
    rfc_ft_test_score = rfc.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {rfc_ft_test_score}')
    print('\n')

    # Gradient Boosting Machines with FastText dataset
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(avg_ft_train_features, train_label_names)
    gbc_ft_cv_scores = cross_val_score(gbc, avg_ft_train_features, train_label_names, cv=5)
    gbc_ft_cv_mean_score = np.mean(gbc_ft_cv_scores)
    print('Gradient Boosting Machines with FastText dataset')
    print(f'CV Accuracy (5-fold): {gbc_ft_cv_scores}')
    print(f'Mean CV Accuracy: {gbc_ft_cv_mean_score}')
    gbc_ft_test_score = gbc.score(avg_ft_test_features, test_label_names)
    print(f'Test Accuracy: {gbc_ft_test_score}')
    print('\n')


# =============================================================================
#   Neural Network with all of datasets
# =============================================================================
    
    # Two-hidden layer neural network with Word2Vec dataset
    mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive',
    early_stopping=True, activation = 'relu', hidden_layer_sizes=(512, 512),
    random_state=42)
    mlp.fit(avg_w2v_train_features, train_label_names)
    svm_ft_test_score = mlp.score(avg_w2v_test_features, test_label_names)
    print('Two-hidden layer neural network with Word2Vec dataset')
    print(f'Test Accuracy: {svm_w2v_test_score}')
    print('\n')
    
    # Two-hidden layer neural network with GloVe dataset
    mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive',
    early_stopping=True, activation = 'relu', hidden_layer_sizes=(512, 512),
    random_state=42)
    mlp.fit(train_glove_features, train_label_names)
    svm_ft_test_score = mlp.score(test_glove_features, test_label_names)
    print('Two-hidden layer neural network with GloVe dataset')
    print(f'Test Accuracy: {svm_glove_test_score}')
    print('\n')
    
    # Two-hidden layer neural network with FastText dataset
    mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive',
    early_stopping=True, activation = 'relu', hidden_layer_sizes=(512, 512),
    random_state=42)
    mlp.fit(avg_ft_train_features, train_label_names)
    svm_ft_test_score = mlp.score(avg_ft_test_features, test_label_names)
    print('Two-hidden layer neural network with FastText dataset')
    print(f'Test Accuracy: {svm_ft_test_score}')
