import nltk
import numpy as np
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

newsgroups_data = fetch_20newsgroups(subset='all')

category = ['alt.atheism', 'sci.space','talk.religion.misc', 'comp.graphics']

newsgroups_data = fetch_20newsgroups(subset='all',remove = ('headers','footers','quotes'),categories=category )

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

updated_data = ''.join(newsgroups_data.data)

import pandas as pd
df_unprocessed = pd.DataFrame.from_records(newsgroups_data, columns =['data'] )

#///////////////////////////////////////////////////////////////////////////////////////////////

df = df_unprocessed

#///////////////////////////////////////////////////////////////////////////////////////////////

#https://stackoverflow.com/questions/44227748/removing-newlines-from-messy-strings-in-pandas-dataframe-cells
df['data'] = df['data'].replace('[\n\t]','', regex=True)

#https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas
df['data'] = df['data'].str.replace('[^\w\s\t]','')

df['data'] = df['data'].str.replace('\d+', '')

df['data'] = df['data'].apply(lambda x: x.lower())

df['data'] = df['data'].apply(lambda x: x.split())

def removeStopWords(tokenized_words):
    filtered_sent=[]
    for word in tokenized_words:
        if word not in stop_words:
            filtered_sent.append(word)
    return filtered_sent

df['data'] = df['data'].apply(removeStopWords)

def getPartsOfSpeech(tokenized_words):
    partsOfSpeech = []
    for word in tokenized_words:
        partsOfSpeech.append(nltk.pos_tag([word]))
    return partsOfSpeech

df['partsOfSpeech'] = df['data'].apply(getPartsOfSpeech)


#????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

import nltk
tags = []
for a in filtered_sent:
    tags.append(nltk.pos_tag(a))


# In[216]:


labels = corpus
df = pd.DataFrame.from_records(filtered_sent, columns=labels)


# In[173]:


# Divide the train and test data set into 70-30
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, newsgroups_data.target, test_size=0.3, random_state=123)


# In[172]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(tokenized_word)
idf = vectorizer.idf_
#print (dict(zip(vectorizer.get_feature_names(), idf)))


# In[ ]:





# In[101]:


def genFunc(y):
    final = []
    for z in gensim.utils.simple_preprocess(y) :
        final.append(z)
    return final


# In[102]:


#https://stackoverflow.com/questions/38739250/how-to-install-gensim-on-windows
import gensim
clean_data = []

for x in newsgroups_data.data:
    clean_data.append(genFunc(x))



# In[213]:


import nltk
nltk.download('averaged_perceptron_tagger')

# nouns=[]
# for item in nltk.pos_tag(tokenized_word):
#   if item[1] in ["NNP","NNS"]:
#     print("noun:", item[0])


# In[214]:


tags = []
for a in filtered_sent:
    tags.append(nltk.pos_tag(a))

# nltk.pos_tag(clean_data)
# sent=clean_data
# # word tokens
# tokenized_word=word_tokenize(sent)
# print(tokenized_word)
# print("part of speech",nltk.pos_tag(tokenized_word))


# In[ ]:


print(tags)


# In[112]:


from nltk.probability import FreqDist
fdist = FreqDist(clean_data)
print(fdist)


# In[19]:


fdist.most_common(20)


# In[20]:


get_ipython().system('pip install matplotlib')


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


# In[110]:


#https://www.geeksforgeeks.org/python-convert-a-nested-list-into-a-flat-list/
# output list
output = []

# function used for removing nested
# lists in python.
def reemovNestings(clean_data):
    for i in clean_data:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)

reemovNestings(clean_data)


# In[120]:


#http://www.nltk.org/howto/collocations.html
import pandas as pd
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
# #trigram_measures = nltk.collocations.TrigramAssocMeasures()
# finder = BigramCollocationFinder.from_words('tokenized_word')
# finder.nbest(bigram_measures.pmi, 20)

## Bigrams
finder = nltk.collocations.BigramCollocationFinder.from_words(output)

# only bigrams that appear 5+ times
finder.apply_freq_filter(2)

# return the 50 bigrams with the highest PMI
print(finder.nbest(bigram_measures.pmi, 20))

bigramPMITable = pd.DataFrame(list(finder.score_ngrams(bigram_measures.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)

print(bigramPMITable)


# In[31]:



import pandas as pd
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(output)
finder.apply_freq_filter(2)
print(finder.nbest(bigram_measures.pmi, 20))
bigramPMITable = pd.DataFrame(list(finder.score_ngrams(bigram_measures.student_t)), columns=['bigram','t']).sort_values(by='t', ascending=False)
print(bigramPMITable)


# In[122]:


#https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a
import pandas as pd
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
# #trigram_measures = nltk.collocations.TrigramAssocMeasures()
# finder = BigramCollocationFinder.from_words('tokenized_word')
# finder.nbest(bigram_measures.pmi, 20)

## Bigrams
finder = nltk.collocations.BigramCollocationFinder.from_words(output)

# only bigrams that appear 5+ times
finder.apply_freq_filter(2)

# return the 50 bigrams with the highest PMI
print(finder.nbest(bigram_measures.pmi, 20))

bigramPMITable = pd.DataFrame(list(finder.score_ngrams(bigram_measures.chi_sq)), columns=['bigram','chi_sq']).sort_values(by='chi_sq', ascending=False)

print(bigramPMITable)


# In[54]:


clean_data[:3386]


# In[52]:


output


# In[113]:


#https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
def to_lowercase(clean_data):
    new_words = []
    for word in clean_data:
        new_word = word.lower()
        new_words.append(new_word)
        return new_words





# In[114]:


#https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
def remove_punctuation(clean_data):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in clean_data:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


# In[115]:


print(clean_data)


# In[116]:


filtered_sent=[]
tokenized_sent = output
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
#print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:",filtered_sent)


# In[117]:


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


# In[68]:


import re
re.sub('[^A-Za-z0-9]+', '','clean_data')


# In[76]:


#listToString = ','.join(clean_data)
values = ''.join(str(clean_data)[1:-1])


# In[77]:


values


# In[118]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

#print("Filtered Sentence:",values)
print("Stemmed Sentence:",stemmed_words)


# In[123]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = filtered_sent
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())


# In[87]:


#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = stemmed_words
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())


# In[ ]:


tf_idf = {}
for i in range(N):
    tokens = processed_text[i]
    counter = Counter(tokens + processed_title[i])
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log(N/(df+1))
        tf_idf[doc, token] = tf*idf

