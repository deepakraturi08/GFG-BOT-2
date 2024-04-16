from flask import Flask,render_template, request
# import google.generativeai as palm
# import os
import pickle
import random
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize 
import torch
from torch.nn.utils.rnn import pad_sequence
# from dotenv import load_dotenv

# load_dotenv()

# palm_api_key =os.environ['PALM_API_KEY']
# palm.configure(api_key=palm_api_key)
#set up flask app
app=Flask(__name__)
@app.route("/")
def home():
  return render_template("index.html")

@app.route("/chatbot",methods=["POST"])

def chatbot():
  user_input=request.form["message"]

  # Models=[m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
  # model=Models[0].name

  prompt=f"User: {user_input}\n MY_BOT: "

  #generate response
  # response=palm.generate_text(
  #     model=model,
  #     prompt=prompt,
  #     stop_sequences=None,
  #     temperature=0,
  #     max_output_tokens=100
  # )
  # bot_response= response.result


# Load the saved model from the pickle file
  with open('modeltemp.pkl', 'rb') as file:
    modeltemp=pickle.load(file)
# Preprocess input data if necessary (e.g., scaling, normalization)
  texts_p=[]
  prediction_input=user_input

#removing punctuation and converting to lowercase
  prediction_input=[letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input=''.join(prediction_input)
  texts_p.append(prediction_input)

#tokenizing and padding
  prediction_input=word_tokenize(prediction_input)
  prediction_input=np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequence(prediction_input, padding='post', padding_value=0)

# Make predictions using the loaded model
  bot_response = modeltemp.predict(prediction_input)

  chat_history=[]
  chat_history.append(f"User:{user_input}\n MY_BOT: {bot_response}")

  return render_template(
    "chatbot.html",
    user_input=user_input,
    bot_response=bot_response,
    chat_history=chat_history
  )

if __name__=="__main__":
  app.run(debug=True)