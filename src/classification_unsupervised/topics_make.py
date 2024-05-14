# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:53:16 2023

@author: adamWolf
"""

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import gensim

import remove_stopwords as rm_stpwrds


def topics_make(chapters, language, num_of_topics):
    """
    Analyzes the topics in the given chapters using Latent Dirichlet Allocation (LDA) and Latent Semantic Indexing (LSI) models.
    
    Parameters:
        chapters (list of str): A list of chapters (documents) to be analyzed for topics.
        language (str): The language of the chapters. It must be one of the supported languages.
        num_of_topics (int): The number of topics to be generated.
    
    Raises:
        AttributeError: If the input chapters are not a list of strings or if the language is not supported.
    """
    #normalization of chapters
    wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = WordNetLemmatizer()
    
    garbage_tokens = ['yes', 'ah', 'sir', 'mr', 'mrs', 'em', 'er']
       
    def normalize_corpus(texts):
        """
        Normalizes the given list of texts by removing stopwords, converting to lowercase,
        lemmatizing words, and filtering out non-alphanumeric tokens.
    
        Args:
            texts (list of str): A list of texts to be normalized.
    
        Returns:
            list of list of str: The normalized texts, where each text is represented as a list of tokens.
    
        """
        norm_texts = []
        for text in texts:
            rm_stpwrds.remove_stopwords(text, language)
            text = text.lower()
            text_tokens = [token.strip() for token in wtk.tokenize(text)]
            # 'has -> ha, was -> wa' tokens was generated as well - solved with pos='v'
            text_tokens = [wnl.lemmatize(token, pos = 'v') for token in text_tokens if not token.isnumeric()]
            text_tokens = [token for token in text_tokens if len(token) > 1]
            text_tokens = list(filter(None, text_tokens))
            text_tokens = [token for token in text_tokens if token not in garbage_tokens]
            if text_tokens:
                norm_texts.append(text_tokens)
            
        return norm_texts
        
    norm_texts = normalize_corpus(chapters)
        
    # handling bi-grams
    bigram = gensim.models.Phrases(norm_texts, min_count = 20, threshold = 20,
    delimiter = '_') # higher threshold fewer phrases.
    bigram_model = gensim.models.phrases.Phraser(bigram)
    
    
    # generate phrases for all of our tokenized chapters and build a vocabulary
    # create a unique term/phrase to number mapping
    norm_corpus_bigrams = [bigram_model[doc] for doc in norm_texts]
    # Create a dictionary representation of the documents.
    dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)

    print('Total Vocabulary Size:', len(dictionary))
    
    # Filter out words that occur fewer than 6 times, or more than 60% of the chapters
    # this filters out recurring topics
    dictionary.filter_extremes(no_below = 6, no_above = 0.6)
    print('Total Filtered Vocabulary Size:', len(dictionary))
    
    #feature engineering with a bag of words model
    # Transforming corpus into bag of words vectors
    bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]

    # # viewing actual terms and their counts
    print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])
    # total chapters in the corpus
    print('\nTotal number of chapters:', len(bow_corpus))
    
    
    #latent semantic indexing (LSI)
    TOTAL_TOPICS = num_of_topics
    lsi_bow = gensim.models.LsiModel(bow_corpus, id2word = dictionary,
                                     num_topics = TOTAL_TOPICS,
                                     onepass = True,
                                     chunksize = 1740, 
                                     power_iters = 1000)    
    
# =============================================================================

# =============================================================================
    
    #Latent Dirichlet Allocation (LDA)
    lda_model = gensim.models.LdaModel(corpus = bow_corpus, id2word = dictionary,
                                       chunksize = 1740, alpha = 'auto',
                                       eta = 'auto', random_state = 42,
                                       iterations = 500, num_topics = TOTAL_TOPICS,
                                       passes = 40, eval_every = None)
    
    print('Latent Dirichlet Allocation (LDA)')
    for topic_id, topic in lda_model.print_topics(num_topics = 5, num_words = 50):
        print('Topic #' + str(topic_id + 1) + ':')
        print(topic)
    
    #overall average coherence score of the model
    topics_coherences = lda_model.top_topics(bow_corpus, topn = 20)
    avg_coherence_score = np.mean([item[1] for item in topics_coherences])
    print('Avg. Coherence Score:', avg_coherence_score)
    
    #LDA topics with weights
    topics_with_wts = [item[0] for item in topics_coherences]
    print('LDA Topics with Weights')
    print('=' * 50)
    for idx, topic in enumerate(topics_with_wts):
        print('Topic #' + str(idx + 1) + ':')
        print([(term, round(wt, 3)) for wt, term in topic])
        
    #LDA topics without weights
    print('\nLDA Topics without Weights')
    print(f'Total topics: {TOTAL_TOPICS}')
    print('=' * 50)
    for idx, topic in enumerate(topics_with_wts):
        print('Topic ' + str(idx + 1) + ':')
        print([term for wt, term in topic])
    