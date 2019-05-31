import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
import en_core_web_sm

from nltk import FreqDist
import numpy as np
import re
import spacy
import gensim
from gensim import corpora
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

pd.set_option("display.max_colwidth", 200)
df = pd.read_csv('C:/Users/rochisman.datta/Desktop/Python Code/Comments Dump/CSV/hdfc-.csv')

def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 2]
    
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = " ".join(text)

    return(text)
# lambda one line: func call
# .apply: passes func on each value of a pandas series
df["Clean_Comments"] = df["Comments"].apply(lambda x: clean_text(x))
df['Clean_Comments'] = df['Clean_Comments'].str.replace("[^a-zA-Z#]", " ")
#function to plot most common words

def freq_words(x, terms = 30):
##    for text in x:
##        if (get_wordnet_pos(x[1]) == 'NNS' or 'NN' or 'NNP' or 'NNPS'):
##            all_nouns = ' '.join(text)

   
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()

nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_reviews = pd.Series(df['Clean_Comments']).apply(lambda x: x.split())

reviews_2 = lemmatization(tokenized_reviews)
##print(reviews_2[1]) # print lemmatized review

reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df['Common_Words'] = reviews_3

##freq_words(df['Common_Words'], 35)

dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100,chunksize=1000, passes=50)

lda_model.print_topics()

vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.save_html(vis, 'LDA_Visualization_HDFC.html')
vis

