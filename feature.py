'''
feature extraction for text data, use mini_batch for online feature extraction, for arbitrary large size data
 sample usage:
    python features.py --input_train_text ../data/Train_200k.csv --output_train_feature train_200k_pixel.tsv --output_train_label train_labels_200k_pixel.tsv --input_test_text ../data/Test_50k.csv  --output_test_feature test_50k_pixel.tsv --output_test_label test_labels_50k_pixel.tsv --output_feature_extractor feature_extractor_pixel.pickle  --output_hdf5 kaggle_data_pixel.h5
'''
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import *
import numpy as np
import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')
import time
from sklearn import preprocessing
import pandas as pd
import h5py
from common import *

class LinguisticVectorizer(BaseEstimator):

    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg',
                         'nouns', 'adjectives', 'verbs', 'adverbs',
                         'allcaps', 'exclamation', 'question'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        # http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        sent = tuple(nltk.word_tokenize(d))
        if poscache is not None:
            if d in poscache:
                tagged = poscache[d]
            else:
                poscache[d] = tagged = nltk.pos_tag(sent)
        else:
            tagged = nltk.pos_tag(sent)

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        for w, t in tagged:
            p, n = 0, 0
            sent_pos_type = None
            if t.startswith("NN"):
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                sent_pos_type = "r"
                adverbs += 1

            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, w)

                if sent_word in sent_word_net:
                    p, n = sent_word_net[sent_word]

            pos_vals.append(p)
            neg_vals.append(n)

        l = len(sent)
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)

        return [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val,
                nouns / l, adjectives / l, verbs / l, adverbs / l]

    def transform(self, documents):
        obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array(
            [self._get_sentiments(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []

        for d in documents:
            allcaps.append(
                np.sum([t.isupper() for t in d.split() if len(t) > 2]))

            exclamation.append(d.count("!"))
            question.append(d.count("?"))

        result = np.array(
            [obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs, allcaps,
             exclamation, question]).T

        return result


emo_repl = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ",  # :D in lower case
    ":dd": " good ",  # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

emo_repl_order = [k for (k_len, k) in reversed(
    sorted([(len(k), k) for k in emo_repl.keys()]))]

re_repl = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\*+": " ", #trim unknown numbers
}


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def preprocessor(tweet):
    tweet = tweet.lower()

    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])
    for r, repl in re_repl.iteritems():
        tweet = re.sub(r, repl, tweet)

    return tweet.replace("-", " ").replace("_", " ")

tfidf_ngrams = StemmedTfidfVectorizer(preprocessor=preprocessor,
                                analyzer="word",
                                ngram_range=(1,2),
                                #min_df=10, max_df=0.5,
                                max_features=1000,
                                stop_words='english', decode_error='ignore'
                                )

def create_union_features(X):
    ling_stats = LinguisticVectorizer()
    all_features = FeatureUnion(
        [('ling', ling_stats), ('tfidf', tfidf_ngrams)])

    return all_features.fit_transform(X)


# === one-hot encoding === #
# we want to encode the category IDs encountered both in
# the training and the test set, so we fit the encoder on both
#encoder = preprocessing.OneHotEncoder()
#encoder.fit(np.vstack((X, X_test)))
#X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
#X_test = encoder.transform(X_test)

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            #print fea
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            features = np.concatenate(extracted, axis=1)
        else:
            features = extracted[0]
        return features

class ImageFeatureMapper:
    def __init__(self, features, vocab="Alphabetic", max_text_len=500):
        self.features = features
        self.vocab = vocab
        self.text_len = max_text_len

    def transform(self,X, y=None):
        extracted = []

        extractor = SemanticPixelEncoder(self.vocab,self.text_len)

        text = None

        for column_name in self.features:
            if(text is None):
                text = X[column_name]
            else:
                text = ['%s %s' % t for t in zip(text, X[column_name])]

        return extractor.transform(text,y)

    def fit_transform(self, X, y=None):
        return self.transform(X,y)

class SemanticPixelEncoder(BaseEstimator, TransformerMixin): #text to binary image conversion
    """Encode any string as a fixed size binary image, column is position, row is character.

    Attributes
    ----------
    `columns` : the maximum length of text we currently support. if longer, we take first 80% and last 20%

    Examples:

    """
    alphabetic = "abcdefghijklmnopqrstuvwxyz"
    numeric = "0123456789"
    separator = "\t\n ,:;.?!-_/$%@&*()'\"#`~^"
    emotion = "= -;:!?/.'\"()@$*&#"
    nonexisting = "\0"

    vocabulary = {
        "Alphabetic": alphabetic,
        "Numeric" : numeric,
        "AlphaNumeric" : alphabetic + numeric,
        "AlphaSeparator" : alphabetic + separator,
        "Twitter" : alphabetic + numeric + emotion,
        "Full" : alphabetic + numeric + separator
    }

    def __init__(self, vocab="Alphabetic", max_text_len=500): #default feature size is 50k
        self.COLUMNS=max_text_len
        self.NONEXISTING= SemanticPixelEncoder.nonexisting

        letters = SemanticPixelEncoder.vocabulary[vocab] +  self.NONEXISTING

        chars = np.array(list(letters.lower())) #convert to np array
        letters = np.unique(chars)

        self.ROWS=len(letters)
        self.dictionary=dict(zip(letters,range(self.ROWS)))

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def normalize(self,text):
        chars = list(text.lower()) #convert to list of lower case characters

        if(len(chars) > self.COLUMNS): #normalize length
            first = int(0.8 * self.COLUMNS)
            last  = int(0.2 * self.COLUMNS)
            chars = chars[0:first] + chars[-last:]

        chars = np.array(chars) #convert to np array
        classes = np.unique(chars)

        if len(np.intersect1d(classes, self.dictionary.keys())) < len(classes):
            diff = np.setdiff1d(classes, self.dictionary.keys())
            for d in diff:
                chars[chars==d] = self.NONEXISTING

        return chars


    def transform(self, X,y=None):
        """Transform text to binary image.

        Parameters
        ----------
            X: str text
        """
        features = tuple()
        for text in X:
            chars = self.normalize(text)

            image = np.zeros((self.ROWS,self.COLUMNS),dtype=np.int)

            for col in range(len(chars)):
                row = self.dictionary[chars[col]]
                image[row][col] = 1

            features = features + (np.hstack(image),)

        return np.vstack(features)

    def inverse_transform(self, img):
        """Transform image back to original text. slow as in O(m*n) time, for test purpose only
        """
        image = img.reshape(self.ROWS,self.COLUMNS)

        ivd = {v: k for k, v in self.dictionary.items()}

        text = ""

        for col in self.COLUMNS:
            for row in self.ROWS:
                if(image[row][col]==1):
                    text += ivd[row]
                    break #there should be only one char in each position

        return text


class MyLabelEncoder(BaseEstimator, TransformerMixin): #add unknown category handling, 1-Of-K encoder
    """Encode labels with value between 0 and n_classes-1.

    Attributes
    ----------
    `classes_` : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    """
    def __init__(self):
        self.NONEXISTING = "LT_UnKnow_Category"

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelEncoder was not fitted yet.")

    def fit(self, X, y=None):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        x = column_or_1d(np.hstack((X,[self.NONEXISTING])), warn=True)
        self.classes_ = np.unique(x)
        #self.classes_.add("LT_UnKnow_Category")
        return self

    def fit_transform(self, X, y=None):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        x = column_or_1d(X, warn=True)
        self.classes_, x = np.unique(x, return_inverse=True)
        return [(0,np.float32(c)) for c in x] #add dummpy key for FeatureMapper above to work

    def replace_nonexist(self,X):
        self._check_fitted()
        classes = np.unique(X)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            for t in diff:
                X[X==t] = self.NONEXISTING

    def transform(self, X):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.replace_nonexist(X)
            #raise ValueError("X contains new labels: %s" % str(diff))
        codes = np.searchsorted(self.classes_, X)
        return [(0,np.float32(c)) for c in codes] #add dummpy key for FeatureMapper above to work

    def inverse_transform(self, X):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        self.replace_nonexist(X)

        x = np.asarray(X)
        return self.classes_[x]

def image_feature_extractor(text_features,vocab="Alphabetic", max_text_len=500):
    combined = ImageFeatureMapper(text_features,vocab,max_text_len)
    return combined


def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', StemmedTfidfVectorizer(preprocessor=preprocessor,
                                analyzer="word",
                                ngram_range=(1,1),
                                min_df=10, max_df=0.8,
                                max_features=5000,
                                stop_words='english', decode_error='ignore',dtype=np.float32
                               )),
                ('Title-Bag of Words', 'Title', StemmedTfidfVectorizer(preprocessor=preprocessor,
                                analyzer="word",
                                ngram_range=(1,1),
                                min_df=10, max_df=0.8,
                                max_features=5000,
                                stop_words='english', decode_error='ignore',dtype=np.float32
                               )),
                ('LocationRaw-Bag of Words', 'LocationRaw', CountVectorizer(max_features=3000,dtype=np.float32)),
                ('SourceName 1-of-K encode', 'SourceName', MyLabelEncoder()),
                ('ContractType 1-of-K encode', 'ContractType', MyLabelEncoder()),
                ('Category 1-of-K encode', 'Category', MyLabelEncoder())
                ]

    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=50,
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=30,
                                                random_state=3465343))]
    return Pipeline(steps)

#assume the labels are integers with lognormal distribution, normalize it to [0,1]
def nouse_norm_labels(train_labels, test_labels=None, _axis=0):
    log_x = np.log(train_labels)
    log_x_mean = log_x.mean(axis=_axis)
    log_x_std = log_x.std(axis=_axis)
    log_x -= log_x_mean
    log_x /= log_x_std #this is z-score

    log_x_min = log_x.min()
    log_x_max = log_x.max()

    log_x -= log_x_min
    log_x /= log_x_max - log_x_min

    if(test_labels is not None):
        log_y = np.log(test_labels)
        log_y -= log_x_mean
        log_y /= log_x_std

        log_y -= log_x_min
        log_y /= log_x_max - log_x_min
        log_y[log_y<0] = 0
        log_y[log_y>1] = 1
        return (log_x, log_y)
    else:
        return log_x

def nouse_un_normalize_labels(train_labels,test_labels, _axis=0):
    log_x = np.log(train_labels)
    log_x_mean = log_x.mean(axis=_axis)
    log_x_std = log_x.std(axis=_axis)
    log_x -= log_x_mean
    log_x /= log_x_std #this is z-score

    log_x_min = log_x.min()
    log_x_max = log_x.max()

    #un_normalize for test_lable
    log_y = test_labels
    log_y *= log_x_max - log_x_min
    log_y += log_x_min
    log_y *= log_x_std
    log_y += log_x_mean
    return np.exp(log_y)

def nouse_normalize(data, mu=None, sigma=None):
    '''
    data normalization function
    data : 2D array, each row is one data
    mu   : 1D array, each element is the mean the corresponding column in data
    sigma: 1D array, each element is the standard deviation of the corresponding
           column in data
    '''
    (m, n) = data.shape
    if mu is None or sigma is None:
        sigma = np.ones(n)
        mu = np.mean(data,0)
        dev = np.std(data,0)
        (x,) = np.nonzero(dev)
        sigma[x] = dev[x]

        mu_rep = np.tile(mu, (m, 1))
        sigma_rep = np.tile(sigma, (m, 1))
        return (data - mu_rep)/sigma_rep, mu, sigma
    else:
        mu_rep = np.tile(mu, (m, 1))
        sigma_rep = np.tile(sigma, (m, 1))
        return (data - mu_rep)/sigma_rep

def nouse_un_normalize(data, mu, sigma):
    '''
    un-normalize the normalized data. This is used for visualization purpose
       data : 2D array, each row is one data
    mu   : 1D array, each element is the mean the corresponding column in data
    sigma: 1D array, each element is the standard deviation of the corresponding
           column in data
    '''
    (m, n) = data.shape
    mu_rep = np.tile(mu, (m, 1))
    sigma_rep = np.tile(sigma, (m, 1))
    return np.multiply(data,sigma_rep) + mu_rep



def remove_constant(data,idx=None):
    '''
      Select columns that are not constant (zero stds)
    '''
    if(idx is None):
        max = np.max(data,0)
        min = np.min(data,0)
        (idx,) = np.nonzero(max-min)
        return data[:,idx],idx
    return data[:,idx]

def extract(input_data,delimiter=',',text_feature_columns=['Text'],label_column=None,output_hdf5='data.h5',vocab="Alphabetic", max_text_len=700):

    train = pd.read_csv(input_data, sep = delimiter,error_bad_lines=False)

    minibatch_size = 50

    # Create the data_stream that parses Reuters SGML files and iterates on
    # documents as a stream.
    minibatch_iterators = iter_minibatch(train, minibatch_size)

    extractor = image_feature_extractor(text_feature_columns,vocab,max_text_len)

    store = None

    print("Extracting training features and save to file")

    # Main loop : iterate on mini-batchs of examples
    for i, (X_train_text, _) in enumerate(minibatch_iterators):

        if(i%10==0):
            print "%d lines processed" % (i*minibatch_size)

        train_data = extractor.fit_transform(X_train_text)


        train_data = pd.DataFrame(train_data)


        if(store is None):
            store = h5py.File(output_hdf5,'w')
            tain_dataset = store.create_dataset('data',data=train_data,dtype='u1', chunks=(1,train_data.shape[1]), maxshape=(None,train_data.shape[1]))
        else:
            start = tain_dataset.shape[0]
            end = tain_dataset.shape[0]+train_data.shape[0]
            tain_dataset.resize(end, 0)
            tain_dataset[start:end] = train_data

    if(store is not None):
        store.close()

    return None if label_column is None else train[label_column]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Given raw text data, generating semantic pixel feature vectors')

    parser.add_argument("-i","--input_data", type=str, help="the tsv file for training",required=True)
    parser.add_argument("-d","--delimiter", type=str, help="the delimiter for the tsv file",default='\t',required=True)

    parser.add_argument("-c", "--text_feature_columns", type=str, nargs='+', help="the text columns in the tsv file for using as features", default=["Text"], required=True)
    parser.add_argument("-o","--output_hdf5", type=str, help="the hdf5 format output file", default='data.h5', required=True)

    args = parser.parse_args()

    print("Command line:\n\tpython %s" % " ".join(sys.argv) )

    print('Arguments:')

    for k,v in args.__dict__.items():
        print('\t%s %s' % (k, v))

    print("Reading in the input text data")

    extract(args.input_data,args.delimiter,args.text_feature_columns,args.output_hdf5)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "Time: " + str(time.time() - start_time)

    #import cProfile
    #cProfile.run('main()')

