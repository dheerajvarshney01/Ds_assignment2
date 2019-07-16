import nltk
import numpy as np
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

category = ['alt.atheism', 'sci.space','talk.religion.misc', 'comp.graphics']

newsgroups_data = fetch_20newsgroups(subset='all',remove = ('headers','footers','quotes'),
                                    categories=category, shuffle = True)

import pandas as pd
df_unprocessed = pd.DataFrame.from_records(newsgroups_data, columns =['data'] )
df_unprocessed = pd.DataFrame(df_unprocessed).join(pd.DataFrame(newsgroups_data.target,
                        columns=['category'], index=pd.DataFrame(df_unprocessed).index))


#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.a
#///////////////////////////////////////////////////////////////////////////////////////////////
df = df_unprocessed
## drop rows where data == '':
df = df.drop(df[df.data == ''].index)

## Remove numbers and other non-letter characters: ###############################################
#https://stackoverflow.com/questions/44227748/removing-newlines-from-messy-strings-in-pandas-dataframe-cells
df['data'] = df['data'].replace('[\n]',' ', regex=True)

#https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas
df['data'] = df['data'].str.replace('[^\'\w\s\t]',' ')

df['data'] = df['data'].str.replace('\d+', '')

df['data'] = df['data'].apply(lambda x: x.lower())

df['data'] = df['data'].apply(lambda x: x.split())

## FRemove stop words: ############################################################################
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

def removeStopWords(tokenized_words):
    filtered_sent=[]
    for word in tokenized_words:
        if word not in stop_words:
            filtered_sent.append(word)
    return filtered_sent

df['data'] = df['data'].apply(removeStopWords)

## Stem the words: ################################################################################
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stemWords(commentList):
    stemmedWords = []
    for word in commentList:
        stemmedWords.append(ps.stem(word))
    return stemmedWords

df['data'] = df['data'].apply(stemWords)



#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.b
#///////////////////////////////////////////////////////////////////////////////////////////////
## First of all let's concatinate tokens to form documents:
df['documents'] = df['data'].apply(lambda x : ' '.join(x))

## convert the corpus into a bag-of-words tf-idf weighted vector representation:
from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer()
sparseMatrixOfCounts = countVectorizer.fit_transform(df['documents'])
print('Vocabulary size = ', len(countVectorizer.vocabulary_.keys()))

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(sparseMatrixOfCounts)

#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.c
#///////////////////////////////////////////////////////////////////////////////////////////////
## Let's add the tfidf to the dataframe:
df = pd.DataFrame(df).join(pd.DataFrame(tfidf.toarray(), index=pd.DataFrame(df).index))

## Drop 'documents' and 'data' columns:
df.drop(['documents', 'data'], axis= 1, inplace=True)

## Split dataframe to the train and test subsets:
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)
##train = train.reset_index().drop(['index'], axis=1)
##test = test.reset_index().drop(['index'], axis=1)
train_x = train.drop('category', axis=1)
train_y = train['category']
test_x = test.drop('category', axis=1)
test_y = test['category']

## Train SVM and report confusion matrix:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
################ 'rbf' kernel:
from sklearn.svm import SVC
clfSVC_rbf = SVC(kernel = 'rbf')
clfSVC_rbf.fit(train_x, train_y)

pred_y_rbf = clfSVC_rbf.predict(test_x)
cm_rbf = confusion_matrix(test_y, pred_y_rbf)
accuracy_rbf = accuracy_score(test_y, pred_y_rbf)

################ 'linear'kernel:
from sklearn.svm import SVC
clfSVC_linear = SVC(kernel = 'linear')
clfSVC_linear.fit(train_x, train_y)

pred_y_linear = clfSVC_linear.predict(test_x)
cm_linear = confusion_matrix(test_y, pred_y_linear)
accuracy_linear = accuracy_score(test_y, pred_y_linear)

################ 'poly'kernel:
from sklearn.svm import SVC
clfSVC_poly = SVC(kernel = 'poly')
clfSVC_poly.fit(train_x, train_y)

pred_y_poly = clfSVC_poly.predict(test_x)
cm_poly = confusion_matrix(test_y, pred_y_poly)
accuracy_poly = accuracy_score(test_y, pred_y_poly)

################ 'sigmoid'kernel:
from sklearn.svm import SVC
clfSVC_sigmoid = SVC(kernel = 'sigmoid')
clfSVC_sigmoid.fit(train_x, train_y)

pred_y_sigmoid = clfSVC_sigmoid.predict(test_x)
cm_sigmoid = confusion_matrix(test_y, pred_y_sigmoid)
accuracy_sigmoid = accuracy_score(test_y, pred_y_sigmoid)

## Train MultinomialNB and report confusion matrix:
from sklearn.naive_bayes import MultinomialNB
clfMNB = MultinomialNB()
clfMNB.fit(train_x, train_y)

pred_y_MNB = clfMNB.predict(test_x)
cm_MNB = confusion_matrix(test_y, pred_y_MNB)
accuracy_MNB = accuracy_score(test_y, pred_y_MNB)


#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.d
#///////////////////////////////////////////////////////////////////////////////////////////////
df_nounsOnly = df_unprocessed
## drop rows where data == '':
df_nounsOnly = df_nounsOnly.drop(df_nounsOnly[df_nounsOnly.data == ''].index)

## Perform part od speech tagging on the data:
df_nounsOnly['tokenized_data'] = df_nounsOnly['data'].apply(lambda x: nltk.pos_tag(x.split()))

## Extract nouns:
def extractNouns(tuples):
    nouns = []
    for tup in tuples:
        if tup[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            nouns.append(tup[0])
    return ' '.join(nouns)


df_nounsOnly['nouns_only'] = df_nounsOnly['tokenized_data'].apply(extractNouns)
## Drop columns unused in training and prediction:
df_nounsOnly.drop(['data', 'tokenized_data'], inplace = True, axis = 1)

## Perform cleaning as in Q2.a:
### Remove numbers and other non-letter characters: ###############################################
#https://stackoverflow.com/questions/44227748/removing-newlines-from-messy-strings-in-pandas-dataframe-cells
df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].replace('[\n]',' ', regex=True)

#https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas
df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].str.replace('[^\'\w\s\t]',' ')

df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].str.replace('\d+', '')

df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].apply(lambda x: x.lower())

df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].apply(lambda x: x.split())

### FRemove stop words: ############################################################################
df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].apply(removeStopWords)
### Stem nouns: ################################################################################
df_nounsOnly['nouns_only'] = df_nounsOnly['nouns_only'].apply(stemWords)
### Convert nouns into joined documents:
df_nounsOnly['documents'] = df_nounsOnly['nouns_only'].apply(lambda x : ' '.join(x))

## convert the corpus into a bag-of-words tf-idf weighted vector representation:
countVectorizer_nounsOnly = CountVectorizer()
sparseMatrixOfCounts = countVectorizer_nounsOnly.fit_transform(df_nounsOnly['documents'])
print('Vocabulary size = ', len(countVectorizer.vocabulary_.keys()))

transformer_nounsOnly = TfidfTransformer(smooth_idf=False)
tfidf_nounsOnly = transformer_nounsOnly.fit_transform(sparseMatrixOfCounts)

## Let's add the tfidf to the dataframe:
df_nounsOnly = pd.DataFrame(df_nounsOnly).join(pd.DataFrame(tfidf_nounsOnly.toarray(), index=pd.DataFrame(df_nounsOnly).index))

## Drop 'documents' and 'nouns_only' columns:
df_nounsOnly.drop(['documents', 'nouns_only'], axis= 1, inplace=True)

## Split dataframe to the train and test subsets:
train_nounsOnly, test_nounsOnly = train_test_split(df_nounsOnly, test_size=0.3)
train_x_nounsOnly = train_nounsOnly.drop('category', axis=1)
train_y_nounsOnly = train_nounsOnly['category']
test_x_nounsOnly = test_nounsOnly.drop('category', axis=1)
test_y_nounsOnly = test_nounsOnly['category']

################ 'rbf' kernel:
clfSVC_rbf_nounsOnly = SVC(kernel = 'rbf')
clfSVC_rbf_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_rbf_nounsOnly = clfSVC_rbf_nounsOnly.predict(test_x_nounsOnly)
cm_rbf_nounsOnly = confusion_matrix(test_y, pred_y_rbf_nounsOnly)
accuracy_rbf_nounsOnly = accuracy_score(test_y, pred_y_rbf_nounsOnly)

################ 'linear'kernel:
clfSVC_linear_nounsOnly = SVC(kernel = 'linear')
clfSVC_linear_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_linear_nounsOnly = clfSVC_linear_nounsOnly.predict(test_x_nounsOnly)
cm_linear_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_linear_nounsOnly)
accuracy_linear_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_linear_nounsOnly)

################ 'poly'kernel:
clfSVC_poly_nounsOnly = SVC(kernel = 'poly')
clfSVC_poly_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_poly_nounsOnly = clfSVC_poly_nounsOnly.predict(test_x_nounsOnly)
cm_poly_nounsOnly = confusion_matrix(test_y, pred_y_poly_nounsOnly)
accuracy_poly_nounsOnly = accuracy_score(test_y, pred_y_poly_nounsOnly)

################ 'sigmoid'kernel:
clfSVC_sigmoid_nounsOnly = SVC(kernel = 'sigmoid')
clfSVC_sigmoid_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_sigmoid_nounsOnly = clfSVC_sigmoid_nounsOnly.predict(test_x_nounsOnly)
cm_sigmoid_nounsOnly = confusion_matrix(test_y, pred_y_sigmoid_nounsOnly)
accuracy_sigmoid_nounsOnly = accuracy_score(test_y, pred_y_sigmoid_nounsOnly)

## Train MultinomialNB and report confusion matrix:
clfMNB_nounsOnly = MultinomialNB()
clfMNB_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_MNB_nounsOnly = clfMNB_nounsOnly.predict(test_x_nounsOnly)
cm_MNB_nounsOnly = confusion_matrix(test_y, pred_y_MNB_nounsOnly)
accuracy_MNB_nounsOnly = accuracy_score(test_y, pred_y_MNB_nounsOnly)
































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

