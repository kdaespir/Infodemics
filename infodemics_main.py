import string
import pandas as pd
import numpy as np
import re



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

    def cleaning(self):
        self.cleaned_text = self.text.str.lower()
        for char in self.cleaned_text:
            if type(char) != type(""):
                continue
            if char in string.punctuation:
                self.cleaned_text = self.cleaned_text.replace(char, "")
        
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'rt @.+? ', value="", regex=True)
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+? ', value="", regex=True)
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+', value="", regex=True)
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"(^ +)", value="", regex=True)

    def finalize(self):
        self.dataset["Tweet"] = self.cleaned_text
        print(self.dataset)




if __name__ == "__main__":
    x = infod_classification()
    x.cleaning()
    x.finalize()