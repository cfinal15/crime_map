from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.schema import Document
import json
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load JSON document summaries
with open("eda_documents.json") as f:
    data = json.load(f)

documents = [
    Document(page_content=d["content"], metadata={"title": d["title"], "tags": d["tags"]})
    for d in data
]

# LangChain RAG setup
embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
db = FAISS.from_documents(documents, embedding)
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=db.as_retriever())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Missing 'message' in request"}), 400

    response = qa.run(user_input)
    return jsonify({"reply": response})
