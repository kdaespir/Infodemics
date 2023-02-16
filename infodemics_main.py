import string
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

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

        self.stemmed_texts = []
        stemmer = WordNetLemmatizer()
        for sentence in self.cleaned_text:
            if type(sentence) == type(float(3.3)):
                print(sentence)
                exit()
            lis_words = sentence.split(" ")
            stemmed = [stemmer.lemmatize(word) for word in lis_words ]
            stemmed = " ".join(stemmed)
            self.stemmed_texts += [stemmed]
        
        self.dataset["Tweet"] = self.stemmed_texts
        # print(len(self.stemmed_texts))

    def vectorize(self):

        vectorizer = CountVectorizer(binary=True, stop_words="english")
        vectorizer.fit(np.array(self.dataset["Tweet"]))
        
    def finalize(self):
        print(self.dataset)
        # self.dataset["Tweet"] = self.cleaned_text
        # print(self.dataset)




if __name__ == "__main__":
    x = infod_classification()
    x.preprocess()
    x.finalize()

    # text = "everything is a family affair these days😭🙈and we wouldnt have it any other way.."
    # print(de_emojify(text))