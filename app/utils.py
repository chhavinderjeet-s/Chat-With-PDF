from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def build_vector_store(chunks, path="faiss_db"):
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local(path)


def load_vector_store(path="faiss_db"):
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
