from Data.data_loader import read_data, vectorizer
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
import pickle

def train_and_save_model(model_path, X_train, y_train):
    # define model
    model = MultinomialNB()
    model.fit(X_train,y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model is trained successfully and saved at path {model_path}")

if __name__ == "__main__":
    data = read_data(Path("Artifacts/Data/train_data.csv"))
    X_train = vectorizer(data['Article'], Path("Artifacts/Encoders/vectorizer.pkl"))
    y_train = data['Category'].values
    train_and_save_model(Path("Artifacts/Model/model.pkl"), X_train, y_train)
    