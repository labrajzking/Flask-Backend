from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import pickle#Initialize the flask App
import numpy as np
app = Flask(__name__)
CORS(app,resources={r"/api/*":{"origins":"*"}})
app.config['CORS HEADERS'] = 'Content-Type'
bot = ChatBot('ChatterBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
)
#trainer = ChatterBotCorpusTrainer(bot)
#trainer.train("chatterbot.corpus.english")
training_data = open(r"C:\Users\oussema\Downloads\Flask-Back-End\data\data.txt").read().splitlines()
training_data_disease= open(r"C:\Users\oussema\Downloads\Flask-Back-End\data\data1.txt").read().splitlines()
training_data_stroke = open(r"C:\Users\oussema\Downloads\Flask-Back-End\data\data2.txt").read().splitlines()

iterator=iter(training_data)
iterator_disease=iter(training_data_disease)
iterator_stroke=iter(training_data_stroke)

model = pickle.load(open('model.pkl', 'rb'))
model_disease = pickle.load(open('disease.pkl', 'rb'))
model_stroke = pickle.load(open('stroke.pkl', 'rb'))

user_info=list()
user_info_stroke=list()
user_info_disease=list()

@app.route("/")
@cross_origin()
def home():    
    return render_template("index.html") 
@app.route("/diabets",methods=["POST"])
@cross_origin()
def get_bot_response_diabets():   
    try:
        userText = request.json
        data=userText['msg']   
        user_info.append(data)
        print (user_info)
        return {"data":str(next(iterator)),"done":False}
        #return str(bot.get_response(userText)) 
    except StopIteration:
        # exception will happen when iteration will over
    
        probas=model.predict_proba([np.array(user_info[1:]).astype(np.int_)])
        user_info.clear()
        return {"data":probas[0][1],"done":True}
@app.route("/stroke",methods=["POST"])
@cross_origin()
def get_bot_response_stroke():   
    try:
        userText = request.json
        data=userText['msg']   
        user_info_stroke.append(data)
        print (user_info_stroke)
        return {"data":str(next(iterator_stroke)),"done":False}
        #return str(bot.get_response(userText)) 
    except StopIteration:
        # exception will happen when iteration will over
        probas=model_stroke.predict_proba([np.array(user_info_stroke[1:]).astype(np.int_)])
        user_info_stroke.clear()
        return {"data":probas[0][1],"done":True}
@app.route("/disease",methods=["POST"])
@cross_origin()
def get_bot_response_disease():   
    try:
        userText = request.json
        data=userText['msg']   
        user_info_disease.append(data)
        print (user_info_disease)
        return {"data":str(next(iterator_disease)),"done":False}
        #return str(bot.get_response(userText)) 
    except StopIteration:
        # exception will happen when iteration will over
    
        probas=model_disease.predict_proba([np.array(user_info_disease[1:]).astype(np.int_)])
        user_info_disease.clear()
        return {"data":probas[0][1],"done":True}    
if __name__ == "__main__":    
    app.run()