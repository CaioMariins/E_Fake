from joblib import load
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

#load dos modelos
vetorizador = load("project_news\API\model\dfi_vectorizer.joblib")
pac = load("project_news\API\model\model_pac_titles.joblib")
svc = load("project_news\API\model\model_svc_titles.joblib")
rfc = load("project_news\API\model\model_clf_titles.joblib")
#carregando o Flask
app = Flask(__name__)


#Homepage:
@app.route('/')
def home():
    return render_template('/home.html')

#Modelos:

@app.route("/model")
def model():
    return render_template("/model.html")


@app.route('/predict', methods = ['POST'])
def predict():
    #request of all inputs
    to_pred = [x for x in request.form.values()]
    
    #data preparing
    to_pred = vetorizador.transform(to_pred)
    
    #predict pac
    pac_predict = pac.predict(to_pred)
    rfc_predict = rfc.predict(to_pred)
    svc_predict = svc.predict(to_pred)

    #getting chances:
    pac_proba = pac._predict_proba_lr(to_pred)[0]
    svc_proba = svc.predict_proba(to_pred)[0]
    rfc_proba = rfc.predict_proba(to_pred)[0]
    return render_template("/predict.html", 
                            pac_result = "predict PAC: {}".format(pac_predict), pac_chance = "Chance: {} %".format(round(pac_proba.max()*100,2)),
                            svc_result = "predict SVC: {}".format(svc_predict), svc_chance = "Chance: {} %".format(round(svc_proba.max()*100,2)),
                            rfc_result = "predict RFC: {}".format(rfc_predict), rfc_chance = "Chance: {} %".format(round(rfc_proba.max()*100,2)))


if __name__ == "__main__":
    app.run()