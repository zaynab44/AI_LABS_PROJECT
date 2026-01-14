from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("datasets/mental_health_qa_dataset.csv")

questions = df["Question"].tolist()
answers = df["Answer"].tolist()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

question_embeddings = model.encode(questions)

@app.route("/", methods=["GET", "POST"])
def chat():
    user_message = ""
    bot_reply = ""

    if request.method == "POST":
        user_message = request.form["message"]

        user_embedding = model.encode([user_message])
        similarities = cosine_similarity(user_embedding, question_embeddings)
        best_match = similarities.argmax()

        bot_reply = answers[best_match]

    return render_template("index.html", user=user_message, response=bot_reply)

if __name__ == "__main__":
    app.run(debug=True)