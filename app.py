from flask import Flask,render_template, url_for ,flash , redirect
import joblib
from flask import request
import numpy as np
import os
import pickle
from werkzeug.utils import secure_filename
from keras.models import load_model
import text_hammer as th

app=Flask(__name__)


def text_preprocessing2(text):
    
    text = str(text).lower()
    text =  th.cont_exp(text)
    text= th.remove_emails(text)
    text= th.remove_html_tags(text)
    text = th.remove_special_chars(text)
    text = th.remove_accented_chars(text)
    text = th.make_base(text) #ran -> run,
    return(text)

# Deep Learning Models

with open('token.pkl','rb')as handle:
    token_new=pickle.load(handle)


from keras_preprocessing.sequence import pad_sequences
  
model = load_model('model.h5')



@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method == 'POST':
        text = str(request.form['text'])
        new_text=text_preprocessing2([text])
        new23=token_new.texts_to_sequences([new_text]) # this converts texts into some numeric sequences 
        new234=pad_sequences(new23,maxlen=300,padding='post') 
        print(new23)
        print(new234)
        y_pred = model.predict(new234)
        print(y_pred)
        
        emotion=(np.argmax(y_pred)) 
        percent=y_pred[0][emotion]*100
        print(emotion)
        

        if(emotion==0):
            output_emotion="The Sentence has a Happy Emotion Sentiment"
        elif(emotion==1):
            output_emotion="The Sentence has a Anger Emotion Sentiment"
        elif(emotion==2):
            output_emotion="The Sentence has a Happy and Love Emotion Sentiment"    
        elif(emotion==3):
            output_emotion="The Sentence has a Sadness Sentiment"
        elif(emotion==4):
            output_emotion="The Sentence has a fear Sentiment"
        elif(emotion==5):
            output_emotion="The Sentence has a Suprise and Excited Sentiment"
        else:
            output_emotion="Unable to classify the Sentence"    

            
        
        
        return render_template("index.html", output=output_emotion,percent=round(percent,3))

if __name__ == "__main__":
    app.run(debug=True)
