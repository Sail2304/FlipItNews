import pandas as pd
from pathlib import Path
# To use Regular Expressions
import re

# To use Natural Language Processing
import nltk

# For tokenization
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')


# To remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

# For lemmetization
from nltk import WordNetLemmatizer
nltk.download('wordnet')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def read_data(path:Path):
    data=pd.read_csv(path)
    return data

def text_preprocessing(sent):
    stop_words=list(stopwords.words("english"))
    
    #removing non-letters
    sent = re.sub('[^a-zA-Z]', ' ', sent)

    #word tokenize
    words = nltk.word_tokenize(sent)

    #removing stopwords
    filtered_sent = [w for w in words if w not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    new_text = [lemmatizer.lemmatize(word) for word in filtered_sent]
    new_text = " ".join(new_text)

    return new_text

def target_encoding(data, target_column,path):
    le = LabelEncoder()
    data[target_column]=le.fit_transform(data[target_column].values)
    with open(path,'wb') as file:
        pickle.dump(le, file)
    return data

def split_data(data,train_data_path, test_data_path):
   train,test = train_test_split(data,
                                 test_size=0.2,
                                 shuffle=True,
                                 stratify=data['Category'],
                                 random_state=42)
   train = pd.DataFrame(train)
   train.to_csv(train_data_path, index=False)
   test = pd.DataFrame(train)
   test.to_csv(test_data_path,index=False)

def vectorizer(data, path):
    tf_idf = TfidfVectorizer()
    X = tf_idf.fit_transform(data).toarray()
    with open(path, "wb") as file:
        pickle.dump(tf_idf, file)
    
    return X
    

if __name__ == "__main__":
    data_path = Path("Artifacts/Data/flipitnews-data.csv")
    data = read_data(data_path)
    data['Article'] = data['Article'].apply(text_preprocessing)
    data = target_encoding(data, 'Category', Path('Artifacts/Encoders/category.pkl'))
    split_data(data, Path("Artifacts/Data/train_data.csv"),Path("Artifacts/Data/test_data.csv"))
    
    






