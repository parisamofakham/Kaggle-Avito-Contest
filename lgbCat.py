#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
#Forked From - https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
random.seed(2018)
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import re
import string

NFOLDS = 5
SEED = 2018

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

def rep999(x):
    try:
        if x == -999:
            return -1
        else:
            return x
    except:
        return x

print("\nData Load Stage")

training = pd.read_csv('../input/avito-demand-prediction/train.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
traindex = training.index
newtrain = pd.read_csv('../input/avito-cmx/new_train_features.csv', header=0)

print("\nData Load Stage")

newtrain=newtrain.apply(rep999)
newtrain["perform_white_analysis"]=pd.cut(newtrain['perform_white_analysis'], range(-5, 100, 5))
newtrain["perform_black_analysis"]=pd.cut(newtrain['perform_black_analysis'], range(-5, 100, 5))
newtrain["image_size"]=pd.cut(newtrain['image_size'], range(-100, 5000, 100))
newtrain["average_pixel_width"]=pd.cut(newtrain['average_pixel_width'], range(-2, 50, 1))
newtrain["get_blurrness_score"]=pd.cut(newtrain['get_blurrness_score'], range(-200,10000, 200))
newtrain[["average_red","average_green","average_blue"]]=newtrain[["average_red","average_green","average_blue"]].apply(lambda x: round(100*x,2))
newtrain["average_red"]=pd.cut(newtrain['average_red'], range(-110, 110, 10))
newtrain["average_green"]=pd.cut(newtrain['average_green'], range(-110, 110, 10))
newtrain["average_blue"]=pd.cut(newtrain['average_blue'], range(-110, 110, 10))
newtrain["rural"]=pd.cut(newtrain['rural'], range(-10, 100, 10))
newtrain["city_population"]=newtrain['city_population'].apply(lambda x: np.log(x+2))
newtrain["reg_Population"]=newtrain['reg_Population'].apply(lambda x: np.log(x+2))

print("\nData Load Stage")

traincount = pd.read_csv('../input/avito-wordcount/traincount.csv', header=0)
training = pd.concat([training.reset_index(drop=True), newtrain, traincount], axis=1)
training.index=traindex

print("\nData Load Stage")

testing = pd.read_csv('../input/avito-demand-prediction/test.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
testdex = testing.index
newtest = pd.read_csv('../input/avito-cmx/new_test_features.csv', header=0)

print("\nData Load Stage")

newtest=newtest.apply(rep999)
newtest["perform_white_analysis"]=pd.cut(newtest['perform_white_analysis'], range(-5, 100, 5))
newtest["perform_black_analysis"]=pd.cut(newtest['perform_black_analysis'], range(-5, 100, 5))
newtest["image_size"]=pd.cut(newtest['image_size'], range(-100, 5000, 100))
newtest["average_pixel_width"]=pd.cut(newtest['average_pixel_width'], range(-2, 50, 1))
newtest["get_blurrness_score"]=pd.cut(newtest['get_blurrness_score'], range(-200,10000, 200))
newtest[["average_red","average_green","average_blue"]]=newtest[["average_red","average_green","average_blue"]].apply(lambda x: round(100*x,2))
newtest["average_red"]=pd.cut(newtest['average_red'], range(-110, 110, 10))
newtest["average_green"]=pd.cut(newtest['average_green'], range(-110, 110, 10))
newtest["average_blue"]=pd.cut(newtest['average_blue'], range(-110, 110, 10))
newtest["rural"]=pd.cut(newtest['rural'], range(-10, 100, 10))
newtest["city_population"]=newtest['city_population'].apply(lambda x: np.log(x+2))
newtest["reg_Population"]=newtest['reg_Population'].apply(lambda x: np.log(x+2))

print("\nData Load Stage")

testcount = pd.read_csv('../input/avito-wordcount/testcount.csv', header=0)
testing = pd.concat([testing.reset_index(drop=True), newtest, testcount], axis=1)
testing.index=testdex

print("\nData Load Stage Finished")

ntrain = training.shape[0]
ntest = testing.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print ("Combine Train and Test")

dfs=[training,testing]
df = pd.concat(dfs,axis=0)

print("concatenated")

del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(df.price.mean(),inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
#df["Weekd of Year"] = df['activation_date'].dt.week
#df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical1 = ["perform_white_analysis","perform_black_analysis","image_size","average_pixel_width","get_blurrness_score","average_red","average_green","average_blue","rural"]
categorical2 = ["descsentiment","titlesentiment","user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3","reg_Time_zone"]
categorical=categorical1+categorical2
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical1:
    df[col] = lbl.fit_transform(df[col].astype(str))
for col in categorical2:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 

# Meta Text Features
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
    df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    df[cols + '_num_alphabets'] = df[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
    df[cols + '_num_alphanumeric'] = df[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
    df[cols + '_num_digits'] = df[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
    
# Extra Feature Engineering
df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=17000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            #max_features=7000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds

# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")

del ridge_preds,vectorizer,ready_df
gc.collect();
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.0175,
    'verbose': 0
}  



# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X, y,
                feature_name=tfvocab,
                categorical_feature = categorical)
del X; gc.collect()
# Go Go Go
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=2250,
    verbose_eval=100
)

print("Model Evaluation Stage")
lgpred = lgb_clf.predict(testing) 

#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgb.csv",index=True,header=True)
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))