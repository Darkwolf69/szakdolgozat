# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:14:05 2023

@author: adamWolf
"""

from normalize_document_for_summarization import normalize_document
import basic_functions
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse.linalg import svds
import networkx

text = (basic_functions.open_text('HP-english_original.txt')).read()

#normalize the text
normalize_corpus = np.vectorize(normalize_document)

# get sentences in the document
sentences = nltk.sent_tokenize(text)

# normalize each sentence in the document
norm_sentences = normalize_corpus(sentences)
#print(norm_sentences[500:600])

# =============================================================================
# 
# =============================================================================

# Non-negative Matrix Factorization (NMF)
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
dt_matrix = tv.fit_transform(norm_sentences)
dt_matrix = dt_matrix.toarray()
vocab = tv.get_feature_names_out()
td_matrix = dt_matrix.T
#print matrix row and column numbers
#print(td_matrix.shape)

df = pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)
#print topics from the NMF model table
# print(df)

# =============================================================================
# 
# =============================================================================

#latent semantic analysis - can be applied as content extract or book insight

#method 1: low-rank Singular Value Decomposition (LSI SVD)
def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

#number of sentences and to write, and number of topics (here it is 1, because
#the book has 1 main plot)
num_sentences = 100
num_topics = 1
u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
#print(u.shape, s.shape, vt.shape)
term_topic_mat, singular_values, topic_document_mat = u, s, vt

#compute the sentence saliency scores for each sentence
salience_scores = np.sqrt(np.dot(np.square(singular_values),
np.square(topic_document_mat)))

#selecting the top sentences
top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
top_sentence_indices.sort()
print(f'Number of sentences: {num_sentences}\n'
      f'Number of topics: {num_topics}')
# construct the document summary
# print('\n'.join(np.array(sentences)[top_sentence_indices]))

# =============================================================================
# 
# =============================================================================

#method 2: TextRank
# it gives a more coherent summarization with 100 sentences
similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
# print(similarity_matrix.shape)
np.round(similarity_matrix, 3)

# build the similarity graph
similarity_graph = networkx.from_numpy_array(similarity_matrix)
# # view the similarity graph
# %matplotlib inline
# plt.figure(figsize=(24, 12))
# networkx.draw_networkx(similarity_graph, node_color='lime')

# compute pagerank scores for all the sentences
scores = networkx.pagerank(similarity_graph)
ranked_sentences = sorted(((score, index) for index, score
in scores.items()), reverse=True)

# get the top sentence indices for our summary
top_sentence_indices = [ranked_sentences[index][1]
for index in range(num_sentences)]
top_sentence_indices.sort()
# construct the document summary
print('\n'.join(np.array(sentences)[top_sentence_indices]))
