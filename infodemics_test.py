import string
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler
from langdetect import detect

def de_emojify(text):
    regex_pattern = re.compile(pattern = "["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u23ea"
                           "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'', text)

class infod_classification:
    def __init__(self) -> None:
        non_miss_data = pd.read_excel("Dataset_final.xlsx", sheet_name="Non-misinformation")
        miss_data = pd.read_excel("Dataset_final.xlsx", sheet_name="Non-misinformation")

        non_miss_data["Label"] = 0
        miss_data["Label"] = 1

        self.dataset = pd.concat([non_miss_data, miss_data], axis=0)
        self.dataset = self.dataset.drop(["Number"], axis=1)
        self.dataset = self.dataset.dropna(axis=0)
        self.text = self.dataset["Tweet"]
        pass

    def engl_only(self):
        eng_text = []
        for tweet in self.dataset["Tweet"]:
            try:
                eng_text += [detect(tweet)]
            except:
                eng_text += [False]
                Exception
        eng_bool = [item == "en" for item in eng_text]
        self.dataset["en_label"] = eng_bool
        self.dataset = self.dataset[self.dataset["en_label"] == True]
        self.text = self.dataset["Tweet"]

    def preprocess(self):
        self.cleaned_text = self.text.str.lower()
        for text in self.cleaned_text:
            text = de_emojify(str(text))
            if type(text) != type(""):
                continue
            if text in string.punctuation:
                self.cleaned_text = self.cleaned_text.replace(text, "")
        
        # Removes all retweets at other users
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'rt @[A-Za-z0-9_]+', value="", regex=True)

        # Removes all @s that are not retweets
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'@[A-Za-z0-9_]+', value="", regex=True)

        # Removes all hashtags from the texts
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'#[A-Za-z0-9_]+', value="", regex=True)
        
        # Removes all links that are found in the beginning or middle of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+? ', value="", regex=True)
        
        # Removes all links that are found at the end of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+', value="", regex=True)

        #removes all single characters from the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\^[a-zA-Z]\s+', value=' ', regex=True)

        # Removes all numeric characters from the texts
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"[0-9]+", value="", regex=True)

        # substitutes multiple blank characters with a single one
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\s+', value=' ', regex=True)
        
        # Removes beginning of text, if the text begins with a blank charcter
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"(^ +)", value="", regex=True)

        # Removes punctuation from text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'[^\w\s]', value="", regex=True)
        
        #Removes Underscores
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'[_]+', value="", regex=True)

        # Coverts all the texts into word tokens
        text_tokens = [word_tokenize(sentence) for sentence in self.cleaned_text]
        
        # removes all stop words from the tokens
        remove_stops = [word for word in text_tokens if not word in stopwords.words()]
        

        # Joins all the tokenized words back into a sentence
        process_words = [" ".join(words) for words in remove_stops]
        
        # replaces the tweets in the dataframe with the processed tweets
        self.dataset["Tweet"] = process_words
        
    def lemmatize(self):
        stemmed_texts = []
        stemmer = WordNetLemmatizer()
        for sentence in self.dataset["Tweet"]:
            lis_words = sentence.split(" ")
            stemmed = [stemmer.lemmatize(word) for word in lis_words ]
            stemmed = " ".join(stemmed)
            stemmed_texts += [stemmed]

        self.dataset["Tweet"] = stemmed_texts
        print(self.dataset)

    def vectorize(self):
        self.tfidf = TfidfVectorizer(binary=True, stop_words="english")
        self.csr_mat = self.tfidf.fit_transform(list(self.dataset["Tweet"]))
        self.words = self.tfidf.get_feature_names_out()
        # print(self.words)
        # print(self.csr_mat)

    def dim_redu(self):
        tfid_array = pd.DataFrame(self.csr_mat.toarray(), columns=self.words)
        svd = TruncatedSVD()
        self.redu_text = svd.fit_transform(self.csr_mat)
        print(tfid_array)

    def splits(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split\
            (self.redu_text, self.dataset["Label"], test_size=0.3, random_state=0)

    def ovrsampl(self):
        ros = RandomOverSampler()
        self.xtrain_ros, self.y_train_ros = (self.xtrain, self.ytrain)

    def knn_class(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.xtrain_ros, self.y_train_ros)
        pred = knn.predict(self.xtest)
        acc = accuracy_score(self.ytest, pred)
        print(acc)



x = infod_classification()
x.engl_only()
x.preprocess()
x.lemmatize()
x.vectorize()
x.dim_redu()
x.splits()
x.ovrsampl()
x.knn_class()

# print("_" in string.punctuation)