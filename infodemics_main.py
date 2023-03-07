import string
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from langdetect import detect

# _ = nltk.download("all")



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
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'rt @.+? ', value="", regex=True)
        
        # Removes all links that are found in the beginning or middle of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+? ', value="", regex=True)
        
        # Removes all links that are found at the end of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+', value="", regex=True)

        #removes all single characters from the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\^[a-zA-Z]\s+', value=' ', regex=True)

        # substitutes multiple blank characters with a single one
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\s+', value=' ', regex=True)
        
        # Removes beginning of text, if the text begins with a blank charcter
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"(^ +)", value="", regex=True)

        self.dataset["Tweet"] = self.cleaned_text
        self.dataset = self.dataset.dropna(axis=0)
        self.cleaned_text = self.cleaned_text.dropna(axis=0)

    def stem(self, stem = "wordnet"):
        
        self.stemmed_texts = []
        if stem == "wordnet":
            stemmer = WordNetLemmatizer()
            for sentence in self.cleaned_text:
                lis_words = sentence.split(" ")
                stemmed = [stemmer.lemmatize(word) for word in lis_words ]
                stemmed = " ".join(stemmed)
                self.stemmed_texts += [stemmed]
        else:
            stemmer = PorterStemmer()
            for sentence in self.cleaned_text:
                lis_words = sentence.split(" ")
                stemmed = [stemmer.stem(word) for word in lis_words]
                stemmed = " ".join(stemmed)
                self.stemmed_texts += [stemmed]

        
        self.dataset["Tweet"] = self.stemmed_texts

    def vectorize(self):
        # vectorizer = CountVectorizer(binary=True, stop_words="english")
        self.tfidf = TfidfVectorizer(binary=True, stop_words="english")
        self.csr_mat = self.tfidf.fit_transform(list(self.dataset["Tweet"]))
        self.words = self.tfidf.get_feature_names_out()
        
    def dim_redu(self):
        self.tfid_array = pd.DataFrame(self.csr_mat.toarray(), columns=self.words)
        svd = TruncatedSVD(random_state=None, n_components=5)
        self.dr_text = svd.fit_transform(self.csr_mat)
        print(self.dr_text)

        pass
    
    
    def splits(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split\
            (self.dr_text, self.dataset["Label"], test_size=0.3, random_state=0)
        
        # self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split\
        #     (self.dataset["Tweet"], self.dataset["Label"], test_size=0.3, random_state=0)#, shuffle=True)
        # self.xtrain_vec = self.tfidf.transform(self.xtrain)
        # self.xtest_vec = self.tfidf.transform(self.xtest)
        training_data = pd.DataFrame(self.xtrain,self.ytrain)
        print(training_data)
    def ovsampl(self):
        ros = RandomOverSampler(random_state=0)
        self.xtrain_ros, self.ytrain_ros = ros.fit_resample(self.xtrain, self.ytrain)

    def pipelines(self):
        svd = TruncatedSVD()
        ros = RandomOverSampler()
        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        knn_pipeline = make_pipeline(svd, knn)
        knn_pipeline.fit(self.csr_mat)
        labels = knn_pipeline.predict(self.csr_mat)
        print(labels)
        
    def knn_model(self, k):
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        knn.fit(self.xtrain, self.ytrain)
        pred = knn.predict(self.xtest)
        model_acc = accuracy_score(self.ytest, pred)
        print(f"KNN, k=3 accuracy is {round(model_acc * 100, 2)}%")

    def svc_model(self):
        model = svm.SVC(kernel="linear", probability=True)
        prob = model.fit(self.xtrain_vec, self.ytrain).predict_proba(self.xtest_vec)
        pred = model.predict(self.xtest_vec)
        model_acc = accuracy_score(self.ytest, pred)
        print(f"SVC model accuracy is {round(model_acc * 100, 2)}%")

    def rf_model(self):
        model = RandomForestClassifier()
        model.fit(self.xtrain_vec,self.ytrain)
        pred = model.predict(self.xtest_vec)
        model_acc = accuracy_score(self.ytest, pred)
        print(f"RF model accuracy is {round(model_acc * 100, 2)}%")

    # def lstm_model(self):
    #     model = Sequential(layers=LSTM)
    def finalize(self):
        pass



if __name__ == "__main__":
    x = infod_classification()
    # x.engl_only()
    x.preprocess()
    x.stem(stem="wordnet")
    x.vectorize()
    x.dim_redu()
    x.splits()
    # x.ovsampl()
    # x.pipelines()
    x.knn_model(3)
    # x.svc_model()
    # x.rf_model()
    # x.finalize()

