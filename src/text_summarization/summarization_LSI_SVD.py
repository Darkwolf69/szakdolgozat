# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:14:05 2023

@author: adamWolf
"""

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

from normalize_document_for_summarization import normalize_document


def summarize_LSI_SVD(text, num_of_sentences, num_of_topics):
    """
    Summarize text using Latent Semantic Indexing with Singular Value Decomposition (LSI-SVD) method.

    This function summarizes the given text using LSI-SVD method. It extracts the most salient sentences
    from the text based on their importance scores computed through the LSI-SVD algorithm.

    Parameters:
        text (str): The input text to be summarized.
        num_of_sentences (int): The number of sentences to include in the summary.
        num_of_topics (int): The number of topics to consider in the LSI-SVD method.

    Returns:
        str: The summarized text.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        #normalize the text
        normalize_corpus = np.vectorize(normalize_document)
        
        # get sentences in the document
        sentences = nltk.sent_tokenize(text)
        
        # normalize each sentence in the document
        norm_sentences = normalize_corpus(sentences)
    
        # Non-negative Matrix Factorization (NMF)
        tv = TfidfVectorizer(min_df = 0., max_df = 1., use_idf = True)
        dt_matrix = tv.fit_transform(norm_sentences)
        dt_matrix = dt_matrix.toarray()
        td_matrix = dt_matrix.T
    
    # latent semantic analysis - can be applied as content extract or book insight
    # Low-rank Singular Value Decomposition (LSI SVD) method
        
        def low_rank_svd(matrix, singular_count = 2):
            """
            Computes the low-rank Singular Value Decomposition (SVD) of a matrix.
        
            Parameters:
                matrix (numpy.ndarray): The matrix to decompose.
                singular_count (int): The number of singular values and vectors to compute.
        
            Returns:
                numpy.ndarray: Left singular vectors.
                numpy.ndarray: Singular values.
                numpy.ndarray: Right singular vectors.
                
            
            Examples:
                >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                >>> u, s, vt = low_rank_svd(matrix, singular_count=2)
                >>> u.shape, s.shape, vt.shape
                ((3, 2), (2,), (2, 3))
        
            """
            u, s, vt = svds(matrix, k = singular_count)
            return u, s, vt
        
        # set number of sentences and topics
        num_sentences = num_of_sentences
        num_topics = num_of_topics
        
        u, s, vt = low_rank_svd(td_matrix, singular_count = num_topics)
        # print(u.shape, s.shape, vt.shape)
        term_topic_mat, singular_values, topic_document_mat = u, s, vt
        
        # remove singular values below threshold
        sv_threshold = 0.5
        min_sigma_value = max(singular_values) * sv_threshold
        singular_values[singular_values < min_sigma_value] = 0
        
        #compute the sentence saliency scores for each sentence
        salience_scores = np.sqrt(np.dot(np.square(singular_values),
                                         np.square(topic_document_mat)))
        
        #selecting the top sentences
        top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
        top_sentence_indices.sort()
        
        # construct and return the document summary
        result_text = '\n'.join(np.array(sentences)[top_sentence_indices])
        
        return result_text
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
