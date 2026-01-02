from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings # Matches helper.py
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# 1. Setup Embeddings and Vector Store
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_kwargs={'k': 3})

# 2. Setup Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# 3. Setup Prompt Template
system_prompt = (
    "You are a professional medical assistant. Use the provided context to answer the question. "
    "If you don't know the answer, say you don't know. Keep it concise.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# 4. Create Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)