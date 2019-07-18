import nltk
import numpy as np
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'sci.space','talk.religion.misc', 'comp.graphics']

newsgroups_data = fetch_20newsgroups(subset='all',remove = ('headers','footers','quotes'),
                                    categories=categories, shuffle = True)

## Read the data and the target into DataFrame for further cleaning:
### https://stackoverflow.com/questions/48769682/how-do-i-convert-data-from-a-scikit-learn-bunch-object-to-a-pandas-dataframe
import pandas as pd
data = np.c_[newsgroups_data.data, newsgroups_data.target]
columns = np.append(['data'], ["target"])
df_unprocessed = pd.DataFrame.from_records(data, columns = columns)

#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.a
#///////////////////////////////////////////////////////////////////////////////////////////////
df = df_unprocessed
## Some cells of the 'data' column are empty strings ''. Let's drop them:
df = df.drop(df[df.data == ''].index)

## Remove numbers and other non-letter characters: ###############################################
### https://stackoverflow.com/questions/44227748/removing-newlines-from-messy-strings-in-pandas-dataframe-cells
df['data'] = df['data'].replace('[\n]',' ', regex=True)

###https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas
df['data'] = df['data'].str.replace('[^\w\s\t]',' ')

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
### https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(smooth_idf=False)
tfidf = tfidfVectorizer.fit_transform(df['documents'])

#///////////////////////////////////////////////////////////////////////////////////////////////
# Q2.c
#///////////////////////////////////////////////////////////////////////////////////////////////
## Split dataframe to the train and test subsets:
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(tfidf, df['target'], test_size=0.3)

## Train SVM and report confusion matrix:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
## We will use the folllowing function in order to optimize the algorithms:
### https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds, _kernel, Cs_list, gammas_list):
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=_kernel), param_grid, cv=nfolds, n_jobs = -1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

### best_params = svc_param_selection(train_x, train_y, 3, 'rbf', Cs_list = Cs, gammas_list = gammas)
################ 'rbf' kernel:
### Using the function svc_param_selection() above we have determined the optimal C and gamma.
### For this kernel, the optimization was done on the range of C and gamma values. The last ranges
### used to fine-tune C and gamma were:
### Cs = [10.2,10.4,10.6,10.8,11,11.2,11.4,11.6,11.8]
### gammas = [0.082,0.084,0.086,0.088,0.09,0.092,0.094,0.096,0.098]
clfSVC_rbf = SVC(kernel = 'rbf', C=10.2, gamma=0.094)
clfSVC_rbf.fit(train_x, train_y)

pred_y_rbf = clfSVC_rbf.predict(test_x)
cm_rbf = confusion_matrix(test_y, pred_y_rbf)
print("SVM 'rbf' kernel confusion matrix is \n", cm_rbf)
accuracy_rbf = accuracy_score(test_y, pred_y_rbf)
print("SVM 'rbf' kernel accuracy is ", accuracy_rbf)

## Train MultinomialNB and report confusion matrix:
from sklearn.naive_bayes import MultinomialNB
### Optimization: the optimal alpha=0.04 was determined using the commented code below:
###alphas = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,
###            0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,
###            0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4]
###for _alpha in alphas:
###    clfMNB = MultinomialNB(alpha=_alpha)
###    clfMNB.fit(train_x, train_y)
###
###   pred_y_MNB = clfMNB.predict(test_x)
###    cm_MNB = confusion_matrix(test_y, pred_y_MNB)
###   accuracy_MNB = accuracy_score(test_y, pred_y_MNB)
###    print("accuracy_MNB ", accuracy_MNB)

clfMNB = MultinomialNB(alpha=0.04)
clfMNB.fit(train_x, train_y)

pred_y_MNB = clfMNB.predict(test_x)
cm_MNB = confusion_matrix(test_y, pred_y_MNB)
print("MultinomialNB confusion matrix is \n", cm_MNB)
accuracy_MNB = accuracy_score(test_y, pred_y_MNB)
print("MultinomialNB accuracy is ", accuracy_MNB)

## We can see that the accuracy of MultinomialNB is 0.859, wherease the accuracy of SVM
## with the defauls kernel ('rbf') is slightly lower, 0.839. These numbers might be lower
## by ~0.02 - 0.03 if simulation ran once again, but the difference of 0.02 between
## the accuracies is preserved.
## Therefore, MultinomialNB has accuracy higher then SVM does with 'rbf' kernel, but those
## accuracies are close. As the accuracies are very close, it is hard to tell
## why MultinomialNB is better then SVM with 'rbf' kernel ,but main reason for
## might be the hyperparameter's tuning. Also it might be that the features, extracted
## from the text, are weakly dependent, which makes MultinomialNB yield accurate predictions.

## Changin kernel of the SVM:
################ 'linear'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [1.03,1.04,1.05,1.06]
### gammas = [0.0,0.0002]
clfSVC_linear = SVC(kernel = 'linear', C=1.04, gamma=0.00001)
clfSVC_linear.fit(train_x, train_y)

pred_y_linear = clfSVC_linear.predict(test_x)
cm_linear = confusion_matrix(test_y, pred_y_linear)
print("SVM 'linear' kernel confusion matrix is \n", cm_linear)
accuracy_linear = accuracy_score(test_y, pred_y_linear)
print("SVM 'linear' kernel accuracy is ", accuracy_linear)

################ 'poly'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [1.1,1.15,1.2,1.25,1.3]
### gammas = [0.9,0.95,1,1.05,1.1]
clfSVC_poly = SVC(kernel = 'poly', C=1.2, gamma=1)
clfSVC_poly.fit(train_x, train_y)

pred_y_poly = clfSVC_poly.predict(test_x)
cm_poly = confusion_matrix(test_y, pred_y_poly)
print("SVM 'poly' kernel confusion matrix is \n", cm_poly)
accuracy_poly = accuracy_score(test_y, pred_y_poly)
print("SVM 'poly' kernel accuracy is ", accuracy_poly)

################ 'sigmoid'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [9.6,9.8,10,10.2,10.4]
### gammas = [0.08,0.09,0.1,0.11,0.12]
clfSVC_sigmoid = SVC(kernel = 'sigmoid', C=10.2, gamma=0.11)
clfSVC_sigmoid.fit(train_x, train_y)

pred_y_sigmoid = clfSVC_sigmoid.predict(test_x)
cm_sigmoid = confusion_matrix(test_y, pred_y_sigmoid)
print("SVM 'sigmoid' kernel confusion matrix is \n", cm_sigmoid)
accuracy_sigmoid = accuracy_score(test_y, pred_y_sigmoid)
print("SVM 'sigmoid' kernel accuracy is ", accuracy_sigmoid)

## From the results above we can see that changing the kernel of SVM yiels accuracies (confusion matrixes)
## similar or lower (more confusing) of the default kernel 'rbf'. Again, these results depend
## on the model fine-tuning with the hyperparameters, so it's likely that all the kernels
## will yield very close accuracies (apart from the 'poly' kernel that consistenly
## yields a significantly lower accuracy)

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
tfidfVectorizer_nounsOnly = TfidfVectorizer(smooth_idf=False)
tfidf_nounsOnly = tfidfVectorizer_nounsOnly.fit_transform(df_nounsOnly['documents'])
print('Vocabulary size part c = ', len(tfidfVectorizer.vocabulary_.keys()))
print('Vocabulary size part d = ', len(tfidfVectorizer_nounsOnly.vocabulary_.keys()))

## Split dataframe to the train and test subsets:
train_x_nounsOnly, test_x_nounsOnly, train_y_nounsOnly, test_y_nounsOnly = \
    train_test_split(tfidf_nounsOnly, df_nounsOnly['target'], test_size=0.3)

## We will use the same technique to optimize the models as we used in part c above:
### best_params = svc_param_selection(train_x, train_y, 3, 'rbf', Cs_list = Cs, gammas_list = gammas)
################ 'rbf' kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [5.32,5.34,5.36,5.38,5.4,5.42,5.44,5.46,5.48]
### gammas = [0.452,0.454,0.456,0.458,0.46,0.462,0.464,0.466,0.468]
clfSVC_rbf_nounsOnly = SVC(kernel = 'rbf', C=5.32, gamma=0.462)
clfSVC_rbf_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_rbf_nounsOnly = clfSVC_rbf_nounsOnly.predict(test_x_nounsOnly)
cm_rbf_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_rbf_nounsOnly)
print("SVM 'rbf' kernel confusion matrix for nouns only is \n", cm_rbf_nounsOnly)
accuracy_rbf_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_rbf_nounsOnly)
print("SVM 'rbf' kernel accuracy for nouns only is ", accuracy_rbf_nounsOnly)

## Train MultinomialNB and report confusion matrix:
### Optimization: the optimal alpha=0.04 was determined using the commented code below:
### alphas = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,
###             0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,
###             0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4]
### for _alpha in alphas:
###     clfMNB = MultinomialNB(alpha=_alpha)
###     clfMNB.fit(train_x_nounsOnly, train_y_nounsOnly)
###
###     pred_y_MNB_nounsOnly = clfMNB.predict(test_x_nounsOnly)
###     accuracy_MNB = accuracy_score(test_y_nounsOnly, pred_y_MNB_nounsOnly)
###     print("accuracy_MNB ", accuracy_MNB)

clfMNB_nounsOnly = MultinomialNB(alpha=0.01)
clfMNB_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_MNB_nounsOnly = clfMNB_nounsOnly.predict(test_x_nounsOnly)
cm_MNB_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_MNB_nounsOnly)
print("MultinomialNB confusion matrix for nouns only is \n", cm_MNB_nounsOnly)
accuracy_MNB_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_MNB_nounsOnly)
print("MultinomialNB accuracy for nouns only is ", accuracy_MNB_nounsOnly)

################ 'linear'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [1.292,1.294,1.296,1.298,1.3,1.302,1.304,1.306,1.308]
### gammas = [0.000001]
clfSVC_linear_nounsOnly = SVC(kernel = 'linear', C=1.298, gamma=0.000001)
clfSVC_linear_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_linear_nounsOnly = clfSVC_linear_nounsOnly.predict(test_x_nounsOnly)
cm_linear_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_linear_nounsOnly)
print("SVM 'linear' kernel confusion matrix for nouns only is \n", cm_linear_nounsOnly)
accuracy_linear_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_linear_nounsOnly)
print("SVM 'linear' kernel accuracy for nouns only is ", accuracy_rbf_nounsOnly)

################ 'poly'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [1.52,1.54,1.56,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.86,1.88]
### gammas = [0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98]
### best_params = svc_param_selection(train_x_nounsOnly, train_y_nounsOnly, 3, 'poly', Cs_list = Cs, gammas_list = gammas)
clfSVC_poly_nounsOnly = SVC(kernel = 'poly', C=1.56, gamma=0.92)
clfSVC_poly_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_poly_nounsOnly = clfSVC_poly_nounsOnly.predict(test_x_nounsOnly)
cm_poly_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_poly_nounsOnly)
print("SVM 'poly' kernel confusion matrix for nouns only is \n", cm_poly_nounsOnly)
accuracy_poly_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_poly_nounsOnly)
print("SVM 'poly' kernel accuracy for nouns only is ", accuracy_poly_nounsOnly)

################ 'sigmoid'kernel:
### The last ranges used to fine-tune C and gamma were:
### Cs = [9.5,9.6,9.7,9.8,9.9]
### gammas = [0.1556,0.1558,0.156]
### best_params = svc_param_selection(train_x_nounsOnly, train_y_nounsOnly, 3, 'sigmoid', Cs_list = Cs, gammas_list = gammas)
clfSVC_sigmoid_nounsOnly = SVC(kernel = 'sigmoid', C=9.7, gamma=0.1556)
clfSVC_sigmoid_nounsOnly.fit(train_x_nounsOnly, train_y_nounsOnly)

pred_y_sigmoid_nounsOnly = clfSVC_sigmoid_nounsOnly.predict(test_x_nounsOnly)
cm_sigmoid_nounsOnly = confusion_matrix(test_y_nounsOnly, pred_y_sigmoid_nounsOnly)
print("SVM 'sigmoid' kernel confusion matrix for nouns only is \n", cm_sigmoid_nounsOnly)
accuracy_sigmoid_nounsOnly = accuracy_score(test_y_nounsOnly, pred_y_sigmoid_nounsOnly)
print("SVM 'sigmoid' kernel accuracy for nouns only is ", accuracy_sigmoid_nounsOnly)

## The vacabulary consisting of nouns only contains 17881 words, wherease the vocabulary of par c
## contains 21493 words.
## Here is the table of occuracies:

##               | part c   | part d
##_______________|__________|___________
## rbf SVC       | 0.833    | 0.825
## MultinomialNB | 0.857    | 0.841
## linear SVC    | 0.824    | 0.825
## poly SVC      | 0.59     | 0.627
## sigmoid SVC   | 0.828    | 0.824

##The accuracies above chanch slightly from run to run of the script,
##but in general - the accuracy on the complete vocabulary is statistically higher
##than on the vacabulary containing nouns only.
##Taking in account that accuracy digrades by 1-2% when text is limited to nouns only,
##it might be benificial to remove every part of speech but nouns from documents before
##applying text classification models we worked in this assignment with.