import numpy as np
from pathlib import Path
import pickle

def preprocess(text,vectorizer_path):
    text = np.array([text])
    with open(vectorizer_path, "rb") as f:
        vectorizer=pickle.load(f)
    vectorized_text = vectorizer.transform(text).toarray()
    return vectorized_text


