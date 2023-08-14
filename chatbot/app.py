from flask import Flask, render_template, request, jsonify
import random
import json
import torch
import speech_recognition as sr
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./pytorchchatbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "./pytorchchatbot/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Srinila"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to handle speech input
def process_speech_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source)
            speech = recognizer.recognize_google(audio, language='en-US')
            print("Recognized speech:", speech)
            return speech

        except sr.UnknownValueError:
            print("Unable to recognize speech.")

# Function to handle text input
def process_text_input():
    return request.form['user_input']
    
def get_response(mode):
    if mode == "1":
        # Text input mode
        user_input = process_text_input()
    else:
        # Speech input mode
        user_input = process_speech_input()

    if user_input:
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = torch.from_numpy(X).to(device).unsqueeze(0)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    return response
        else:
            return "I do not understand..."
    else:
        return "I do not understand..."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    mode = request.form['mode']
    response = get_response(mode)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)