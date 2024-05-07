# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:53:16 2023

@author: adamWolf
"""

import remove_stopwords as rm_stpwrds
import nltk
from nltk.stem import WordNetLemmatizer
import gensim
import numpy as np


def make_topics(chapters, language, num_of_topics):

    #normalization of chapters
    wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = WordNetLemmatizer()
    
    garbage_tokens = ['yes', 'ah', 'sir', 'mr', 'mrs', 'em', 'er']
       
    def normalize_corpus(texts):
        norm_texts = []
        for text in texts:
            rm_stpwrds.remove_stopwords(text, language)
            text = text.lower()
            text_tokens = [token.strip() for token in wtk.tokenize(text)]
            # 'has -> ha, was -> wa' tokens was generated as well - solved with pos='v'
            text_tokens = [wnl.lemmatize(token, pos='v') for token in text_tokens if not token.isnumeric()]
            text_tokens = [token for token in text_tokens if len(token) > 1]
            text_tokens = list(filter(None, text_tokens))
            text_tokens = [token for token in text_tokens if token not in garbage_tokens]
            if text_tokens:
                norm_texts.append(text_tokens)
            
        return norm_texts
        
    norm_texts = normalize_corpus(chapters)
    # print(len(norm_texts))
    # print(norm_texts)
    # print('\n')
    
    
    # handling bi-grams
    bigram = gensim.models.Phrases(norm_texts, min_count=20, threshold=20,
    delimiter='_') # higher threshold fewer phrases.
    bigram_model = gensim.models.phrases.Phraser(bigram)
    # generated phrases with gensim bigram model
    # print(bigram_model[norm_texts[6]][:50])
    
    
    # generate phrases for all of our tokenized chapters and build a vocabulary
    # create a unique term/phrase to number mapping
    norm_corpus_bigrams = [bigram_model[doc] for doc in norm_texts]
    # Create a dictionary representation of the documents.
    dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
    # print('Sample word to number mappings:', list(dictionary.items())[:15])
    # print('\n')
    print('Total Vocabulary Size:', len(dictionary))
    
    # Filter out words that occur fewer than 6 times, or more than 60% of the chapters
    # this filters out recurring topics
    dictionary.filter_extremes(no_below=6, no_above=0.6)
    print('Total Filtered Vocabulary Size:', len(dictionary))
    
    
    
    
    #feature engineering with a bag of words model
    # Transforming corpus into bag of words vectors
    bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
    # print(bow_corpus[1][:50])
    # # viewing actual terms and their counts
    # print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])
    # total chapters in the corpus
    print('\nTotal number of chapters:', len(bow_corpus))
    
    
    #latent semantic indexing (LSI)
    TOTAL_TOPICS = num_of_topics
    lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary,
                                     num_topics=TOTAL_TOPICS,
                                     onepass=True,
                                     chunksize=1740, 
                                     power_iters=1000)
    
    # =============================================================================
    # # latent semantic indexing topics
    # print('latent semantic indexing topics')
    # for topic_id, topic in lsi_bow.print_topics(num_topics=5, num_words=20):
    #     print('Topic #'+str(topic_id+1)+':')
    #     print(topic)
    #     print()
    # =============================================================================
    
    #separate the positive and the negative weights
    # print('separate the positive and the negative weights')
    # for n in range(TOTAL_TOPICS):
    #     print('Topic #'+str(n+1)+':')
    #     print('='*50)
    #     d1 = []
    #     d2 = []
    #     for term, wt in lsi_bow.show_topic(n, topn=20):
    #         if wt >= 0:
    #             d1.append((term, round(wt, 3)))
    #         else:
    #             d2.append((term, round(wt, 3)))
    #     print('Direction 1:', d1)
    #     print('-'*50)
    #     print('Direction 2:', d2)
    #     print('-'*50)
    #     print()
    
# =============================================================================

# =============================================================================
    
    #Latent Dirichlet Allocation (LDA)
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary,
                                       chunksize=1740, alpha='auto',
                                       eta='auto', random_state=42,
                                       iterations=500, num_topics=TOTAL_TOPICS,
                                       passes=40, eval_every=None)
    
    # print('Latent Dirichlet Allocation (LDA)')
    # for topic_id, topic in lda_model.print_topics(num_topics=5, num_words=50):
    #     print('Topic #'+str(topic_id+1)+':')
    #     print(topic)
    #     print()
    
    #overall average coherence score of the model
    topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
    avg_coherence_score = np.mean([item[1] for item in topics_coherences])
    # print('Avg. Coherence Score:', avg_coherence_score)
    
    #LDA topics with weights
    topics_with_wts = [item[0] for item in topics_coherences]
    # print('LDA Topics with Weights')
    # print('='*50)
    # for idx, topic in enumerate(topics_with_wts):
    #     print('Topic #'+str(idx+1)+':')
    #     print([(term, round(wt, 3)) for wt, term in topic])
    #     print()
        
    #LDA topics without weights
    print('\nLDA Topics without Weights')
    print(f'Total topics: {TOTAL_TOPICS}')
    print('='*50)
    for idx, topic in enumerate(topics_with_wts):
        print('Topic '+str(idx+1)+':')
        print([term for wt, term in topic])
        print()
    
        
    # TODO: not working yet, maybe there is 
    # cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
    #                                                       corpus=bow_corpus,
    #                                                       texts=norm_corpus_bigrams,
    #                                                       dictionary=dictionary,
    #                                                       coherence='c_v')
    # avg_coherence_cv = cv_coherence_model_lda.get_coherence()
    # print('Avg. Coherence Score (Cv):', avg_coherence_cv)
    
    # umass_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
    #                                                           corpus=bow_corpus,
    #                                                           texts=norm_corpus_bigrams,
    #                                                           dictionary=dictionary,
    #                                                           coherence='u_mass')
    # avg_coherence_umass = umass_coherence_model_lda.get_coherence()
    
    # perplexity = lda_model.log_perplexity(bow_corpus)
    
    # print('Avg. Coherence Score (Cv):', avg_coherence_cv)
    # print('Avg. Coherence Score (UMass):', avg_coherence_umass)
    # print('Model Perplexity:', perplexity)
    
