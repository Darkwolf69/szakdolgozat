# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 02:40:49 2024

@author: adamWolf
"""

import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

def classification_advanced():
    # reading prepared train and test corpus
    with open('datasets_list.npy', 'rb') as f:
        train_corpus = np.load(f, allow_pickle=True)
        test_corpus = np.load(f, allow_pickle=True)
        train_label_nums = np.load(f, allow_pickle=True)
        test_label_nums = np.load(f, allow_pickle=True)
        train_label_names = np.load(f, allow_pickle=True)
        test_label_names = np.load(f, allow_pickle=True)
        
    with open('datasets_list_Word2Vec.npy', 'rb') as f:
        avg_wv_train_features = np.load(f, allow_pickle=True)
        avg_wv_test_features = np.load(f, allow_pickle=True)
        
    with open('datasets_list_GloVe.npy', 'rb') as f:
        train_glove_features = np.load(f, allow_pickle=True)
        test_glove_features = np.load(f, allow_pickle=True)
        
    with open('datasets_list_FastText.npy', 'rb') as f:
        avg_ft_train_features = np.load(f, allow_pickle=True)
        avg_ft_test_features = np.load(f, allow_pickle=True)
        
        
    # SVM-SGD model with Word2Vec dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(avg_wv_train_features, train_label_names)
    svm_w2v_cv_scores = cross_val_score(svm, avg_wv_train_features, train_label_names, cv=5)
    svm_w2v_cv_mean_score = np.mean(svm_w2v_cv_scores)
    print('\n')
    print('SVM-SGD model with Word2Vec dataset')
    print('CV Accuracy (5-fold):', svm_w2v_cv_scores)
    print('Mean CV Accuracy:', svm_w2v_cv_mean_score)
    svm_w2v_test_score = svm.score(avg_wv_test_features, test_label_names)
    print('Test Accuracy:', svm_w2v_test_score)
    print('\n')
    
    # SVM-SGD model with GloVe dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(train_glove_features, train_label_names)
    svm_glove_cv_scores = cross_val_score(svm, train_glove_features, train_label_names, cv=5)
    svm_glove_cv_mean_score = np.mean(svm_glove_cv_scores)
    print('SVM-SGD model with GloVe dataset')
    print('CV Accuracy (5-fold):', svm_glove_cv_scores)
    print('Mean CV Accuracy:', svm_glove_cv_mean_score)
    svm_glove_test_score = svm.score(test_glove_features, test_label_names)
    print('Test Accuracy:', svm_glove_test_score)
    print('\n')
    
    # SVM-SGD model with FastText dataset
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(avg_ft_train_features, train_label_names)
    svm_ft_cv_scores = cross_val_score(svm, avg_ft_train_features, train_label_names, cv=5)
    svm_ft_cv_mean_score = np.mean(svm_ft_cv_scores)
    print('SVM-SGD model with FastText dataset')
    print('CV Accuracy (5-fold):', svm_ft_cv_scores)
    print('Mean CV Accuracy:', svm_ft_cv_mean_score)
    svm_ft_test_score = svm.score(avg_ft_test_features, test_label_names)
    print('Test Accuracy:', svm_ft_test_score)
    print('\n')
    
    # Two-hidden layer neural network with FastText dataset
    mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive',
    early_stopping=True, activation = 'relu', hidden_layer_sizes=(512, 512),
    random_state=42)
    mlp.fit(avg_ft_train_features, train_label_names)
    svm_ft_test_score = mlp.score(avg_ft_test_features, test_label_names)
    print('Two-hidden layer neural network with FastText dataset')
    print('Test Accuracy:', svm_ft_test_score)
    
    
    


classification_advanced()