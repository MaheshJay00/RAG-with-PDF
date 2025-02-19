import streamlit as st
from langchain_community.document_loaders import PDFlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InmemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are a chatbot assistant for question-answering task. Use the following piece of information of retrieved context to answer the question. If you don't know the answer, reply with 'I don't know'. Keep the answers below 5 sentences.
Question: {question}
Context: {context}
Answer:
"""

pdfs_directory = 'RAG-WITH-PDF/pdfs/'

embeddings = OllamaEmbeddings(model = 'deepseek-r1:14b')
vector_store = InmemoryVectorStore(embeddings)

model = OllamaLLM(model = 'deepseek-r1:14b')

def upload_pdf(file):
    with open(pdfs_directory + file.name,'wb') as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    return text_splitter.split_documents(documents)

def index_docuemnts(docuemnts):
    vector_store.add_documents(docuemnts) 

def retrienve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, answer):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt|model

    return chain.invoke({'querstion':question,'context':context})

uploaded_file = st.file_uploader(
    'upload PDF',type ='PDF',secret_multiple_files = False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split.text(documents)
    index_docuemnts(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message('user').write(question)
        related_documents = retrienve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message()