from flask import Flask
from flask import Flask, render_template, request

from langchain.text_splitter import CharacterTextSplitter #text splitter
from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
import os
import requests
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = './pdfs/'
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            save_directory = './pdfs/'
            
            # Create the directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            save_path = os.path.join(save_directory, pdf_file.filename)
            pdf_file.save(save_path)

        else:
            return '<h1>error</h1>'
        loaders = [UnstructuredPDFLoader(os.path.join(save_directory, fn)) for fn in os.listdir(save_directory)]

        index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)
        
        #Load llm with selected one
        llm2=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
        #Prepare the pipeline
        from langchain.chains import RetrievalQA
        chain = RetrievalQA.from_chain_type(llm=llm2, 
                                            chain_type="stuff", 
                                            retriever=index.vectorstore.as_retriever(), 
                                            input_key="question")
        answer = chain.run(question)
        return render_template('index.html', question=question, answer=answer)
        
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run()
