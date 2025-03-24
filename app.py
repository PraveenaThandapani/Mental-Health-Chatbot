from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset (ensure your dataset is in the correct path)
df = pd.read_csv('chatbot_dataset.csv')

# Initialize vectorizer and fit it on your dataset's questions column
vectorizer = TfidfVectorizer()
vectorizer.fit(df['questions'])

@app.route('/')
def home():
    return render_template('chatbot.html')  # Ensure 'chatbot.html' is in the templates folder

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')  # Get message from the client-side
    response = get_best_answer(user_message)
    return jsonify({'response': response})

def get_best_answer(user_input):
    # Transform the user input using the vectorizer
    user_input_vec = vectorizer.transform([user_input])
    
    # Compute cosine similarity between user input and dataset questions
    cosine_similarities = cosine_similarity(user_input_vec, vectorizer.transform(df['questions']))
    
    # Find the best match index based on similarity scores
    best_match_index = cosine_similarities.argmax()
    
    return df['answers'][best_match_index]  # Return the corresponding answer

if __name__ == '__main__':
    app.run(debug=True)
