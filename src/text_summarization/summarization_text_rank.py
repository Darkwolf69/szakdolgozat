# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:29:10 2023

@author: adamWolf
"""

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx

from normalize_document_for_summarization import normalize_document


def summarize_text_rank(text, num_of_sentences):
    """
    Summarize text using the TextRank algorithm.

    This function summarizes the given text using the TextRank algorithm. It identifies the most important
    sentences in the text based on their similarity scores computed using the TextRank algorithm.

    Parameters:
        text (str): The input text to be summarized.
        num_of_sentences (int): The number of sentences to include in the summary.

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
       
        # get the NMF model matrix
        vocab = tv.get_feature_names_out()
        td_matrix = dt_matrix.T
        df = pd.DataFrame(np.round(td_matrix, 2), index = vocab).head(10)


    # latent semantic analysis - can be applied as content extract or book insight
    # TextRank method
    
        # set number of sentences and topics
        num_sentences = num_of_sentences
        
        similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
        np.round(similarity_matrix, 3)
        
        # build the similarity graph
        similarity_graph = networkx.from_numpy_array(similarity_matrix)
        
        # compute pagerank scores for all the sentences
        scores = networkx.pagerank(similarity_graph)
        ranked_sentences = sorted(((score, index) for index, score
                                   in scores.items()), reverse=True)
        
        # get the top sentence indices for our summary
        top_sentence_indices = [ranked_sentences[index][1]
                                for index in range(num_sentences)]
        top_sentence_indices.sort()
        
        # construct and return the document summary
        result_text = '\n'.join(np.array(sentences)[top_sentence_indices])
        
        return result_text
