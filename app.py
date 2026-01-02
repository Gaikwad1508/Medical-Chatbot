from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# --- 1. Setup Embeddings & Retriever ---
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_kwargs={'k': 3})

# --- 2. Setup Groq LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# --- 3. Make Retriever "History Aware" ---
# This system prompt tells the LLM to rewrite the user's question based on history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# This special retriever will now "rewrite" questions before searching Pinecone
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# --- 4. Setup QA Chain (The Answerer) ---
system_prompt = (
    "You are a professional medical assistant. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, say that you don't know. "
    "Keep the answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"), # This passes history to the answerer too
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# --- 5. Combine Chains ---
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 6. Chat History Storage ---
# (Simple in-memory list for this session)
chat_history = []

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    
    # 1. Run the RAG chain with the history
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })
    
    answer = response["answer"]
    
    # 2. Update the history for the next turn
    chat_history.append(HumanMessage(content=msg))
    chat_history.append(AIMessage(content=answer))
    
    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)