# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 01:25:57 2024

@author: adamWolf

Preprocesses the data for advanced classification models.
Reads prepared train and test corpus, performs feature engineering with Word2Vec, GloVe, and FastText models,
and saves the resulting features into files.

Returns:
    None

Gensim 4 update info: https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4
"""

import numpy as np
import spacy
import gensim
from nltk.tokenize.toktok import ToktokTokenizer
from gensim.models.fasttext import FastText


def prep_classification_advanced():
    # reading prepared train and test corpus
    with open('datasets_list.npy', 'rb') as f:
        train_corpus = np.load(f, allow_pickle=True)
        test_corpus = np.load(f, allow_pickle=True)

    def document_vectorizer(corpus, model, num_features):
        """
        Convert a corpus of tokenized sentences into document vectors using word embeddings.
    
        Parameters:
            corpus (list of lists): The tokenized sentences to be converted into document vectors.
            model: The pre-trained word embedding model used for generating word vectors.
            num_features (int): The dimensionality of the word vectors.
    
        Returns:
            numpy.ndarray: An array of document vectors.
    
        """
        # updates in Gensim 4: index2word -> index_to_key
        vocabulary = set(model.wv.index_to_key)
        
        def average_word_vectors(words, model, vocabulary, num_features):
            """
            Calculate the average word vector for a list of words.
        
            Parameters:
                words (list): List of words.
                model: The pre-trained word embedding model used for generating word vectors.
                vocabulary (set): Set of words in the vocabulary of the word embedding model.
                num_features (int): The dimensionality of the word vectors.
        
            Returns:
                numpy.ndarray: The average word vector.
        
            """
            feature_vector = np.zeros((num_features,), dtype = "float64")
            nwords = 0.
            for word in words:
                if word in vocabulary:
                    nwords = nwords + 1.
                    feature_vector = np.add(feature_vector, model.wv[word])
            if nwords:
                feature_vector = np.divide(feature_vector, nwords)
            return feature_vector
        
        features = [average_word_vectors(tokenized_sentence, model, vocabulary,
                                         num_features) for tokenized_sentence in corpus]
        return np.array(features)
    
    
# =============================================================================
# feature engineering with Word2Vec
# =============================================================================
    
    # tokenize corpus
    tokenizer = ToktokTokenizer()
    
    tokenized_train = [tokenizer.tokenize(text) for text in train_corpus]
    tokenized_test = [tokenizer.tokenize(text) for text in test_corpus]
    
    # generate word2vec word embeddings
    
    # build word2vec model
    w2v_num_features = 1000
    # updates in Gensim 4: size -> vector-size, iter -> epochs
    w2v_model = gensim.models.Word2Vec(tokenized_train, vector_size = w2v_num_features,
                                       window = 100, min_count = 2, sample = 1e-3, sg = 1, epochs = 5, workers = 10)
    
    # generate document level embeddings
    # we only use train dataset vocabulary embeddings
    # so that test dataset truly remains an unseen dataset
    # generate averaged word vector features from word2vec model
    avg_wv_train_features = document_vectorizer(corpus = tokenized_train,
                                                model = w2v_model, num_features = w2v_num_features)
    avg_wv_test_features = document_vectorizer(corpus = tokenized_test,
                                               model = w2v_model, num_features = w2v_num_features)

    with open('datasets_list_Word2Vec.npy', 'wb') as f:
        np.save(f, avg_wv_train_features)
        np.save(f, avg_wv_test_features)
        
        
# =============================================================================
# feature engineering with GloVe
# =============================================================================
    nlp = spacy.load('en_core_web_sm')
    
    # feature engineering with GloVe model
    train_nlp = [nlp(item) for item in train_corpus]
    train_glove_features = np.array([item.vector for item in train_nlp])
    
    test_nlp = [nlp(item) for item in test_corpus]
    test_glove_features = np.array([item.vector for item in test_nlp])
   
    with open('datasets_list_GloVe.npy', 'wb') as f:
        np.save(f, train_glove_features)
        np.save(f, test_glove_features)
        
    
 # =============================================================================
 # feature engineering with FastText
 # =============================================================================   
    
    ft_num_features = 1000
    # sg decides using skip-gram model (1) or CBOW (0)
    ft_model = FastText(tokenized_train, vector_size = ft_num_features, window = 100,
    min_count = 2, sample = 1e-3, sg = 1, epochs = 5, workers = 10)
    # generate averaged word vector features from word2vec model
    avg_ft_train_features = document_vectorizer(corpus = tokenized_train,
    model = ft_model, num_features=ft_num_features)
    avg_ft_test_features = document_vectorizer(corpus = tokenized_test,
    model = ft_model, num_features = ft_num_features)
    
    with open('datasets_list_FastText.npy', 'wb') as f:
        np.save(f, avg_ft_train_features)
        np.save(f, avg_ft_test_features)