from utils.preprocessing import preprocess
import pickle
from pathlib import Path

def load_model(model_path):
    with open(model_path,"rb") as f:
        model = pickle.load(f)

    return model

def predict(model, text, target_encoder_path):

    vectorized_text = preprocess(text, Path("Artifacts/Encoders/vectorizer.pkl"))
    prediction = model.predict(vectorized_text)
    with open(target_encoder_path,"rb") as f:
        encoder=pickle.load(f)

    decoded_prediction=encoder.inverse_transform(prediction)[0]

    return decoded_prediction

if __name__=="__main__":
    model = load_model(Path("Artifacts/Model/Model.pkl"))
    text = """iaaf contest greek decision international association athletics 
                federation appeal acquittal greek athlete kostas kenteris katerina thanou 
                high profile duo cleared doping offence greek athletics federation segas last 
                month iaaf lodge appeal court arbitration sport ca suspended athlete iaaf statement said doping 
                review board concluded decision erroneous statement continued athlete case refered arbitration ca 
                decision case final binding segas said iaaf appeal expected understandable going await final result 
                prejudge said segas chairman vassilis sevastis kenteris olympic gold olympics thanou 
                suspended iaaf last december failing take routine drug test athens game however independent 
                tribunal overturned ban clearing sprinter avoiding test tel aviv chicago athens failing 
                notify anti doping official whereabouts olympics kenteris lawyer gregory ioannidis described 
                iaaf decision unexpected told bbc sport expect organisation take action quickly iaaf timing extremely 
                surprising creates concern question said ioannidis iaaf yet received complete file proceeding include statement 
                testimony closing speech defence counsel nine audio tape evidence time world discovered truth kenteris thanou dramatically 
                withdrew last summer olympics missing drug test olympic village august pair spent four day hospital claiming injured motorcycle crash 
                international olympic committee demanded iaaf investigate affair thanou kenteris still face 
                criminal trial later year allegedly avoiding test faking motorcycle accident"""
    pred = predict(model, text, Path("Artifacts\Encoders\category.pkl"))
    print(pred)