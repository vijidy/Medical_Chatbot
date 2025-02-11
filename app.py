import os
from flask import Flask,render_template,jsonify,request, redirect, url_for, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from dotenv import load_dotenv
from src.prompt import *

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

embeddings=download_hugging_face_embeddings()

index_name = "medicalbot"


docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever= docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1000,
  "response_mime_type": "text/plain",
}

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",generation_config=generation_config)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain= create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("response : ",response['answer'])
    return str(response['answer'])


# @app.route("/")
# def index():
#     if 'chat_history' not in session:
#         session['chat_history'] = []
#     return render_template('chat.html', chat_history=session['chat_history'])

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form['msg']
    
#     # Add user message to history
#     session['chat_history'] = session.get('chat_history', [])
#     session['chat_history'].append({
#         'text': msg,
#         'type': 'user',
#         'time': datetime.now().strftime('%H:%M')
#     })
    
#     # Get bot response
#     response = rag_chain.invoke({"input": msg})
    
#     # Add bot response to history
#     session['chat_history'].append({
#         'text': response['answer'],
#         'type': 'bot',
#         'time': datetime.now().strftime('%H:%M')
#     })
    
#     session.modified = True
#     return redirect(url_for('index'))







if __name__ =='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
