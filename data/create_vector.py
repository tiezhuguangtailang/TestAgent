from langchain_community.document_loaders  import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from langchain_community.document_loaders.word_document import Docx2txtLoader
import pickle

rag_path ="PATH/TO/data/source/source公积金/公积金_缴存.pdf"
loader = PyPDFLoader(rag_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0, add_start_index=True, separators=['\n \n'])
split_docs1 = text_splitter.split_documents(docs)

rag_path ='PATH/TO/data/source/source公积金/公积金使用（提取）.docx'
loader = Docx2txtLoader(rag_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0, add_start_index=True, separators=['\n\n\n'])
split_docs2 = text_splitter.split_documents(docs)

rag_path ='PATH/TO/data/source/source公积金/公积金使用（贷款）.docx'
loader = Docx2txtLoader(rag_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True, separators=['\n\n\n'])
split_docs3 = text_splitter.split_documents(docs)

rag_path ='PATH/TO/data/source/source公积金/个人认证.docx'
loader = Docx2txtLoader(rag_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True, separators=['\n\n\n'])
split_docs4 = text_splitter.split_documents(docs)

split_docs = split_docs1 + split_docs2 + split_docs3 + split_docs4
print(f"Split into {len(split_docs)} chunks.")

embeddings=HuggingFaceBgeEmbeddings(model_name='PATH/TO/data/bge-m3')

# 加载数据库
persist_directory = "PATH/TO/data/retrieve_vector"
os.makedirs(persist_directory,exist_ok=True)
print("Creating vector store...")
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  
)
print("Vector store created.")

# 将加载的向量数据库持久化到磁盘上
print("Persisting vector store to disk...")
vectordb.persist()
print("Vector store persisted.")

# 创建BM25检索器
print("Creating BM25 retriever...")
bm25retriever = BM25Retriever.from_documents(split_docs)

# BM25Retriever序列化到磁盘
bm25retriever_path='PATH/TO/data/bm25retriever'
if not os.path.exists(bm25retriever_path):
    os.mkdir(bm25retriever_path)

print("Serializing BM25 retriever...")
pickle.dump(bm25retriever, open('PATH/TO/data/bm25retriever/bm25retriever.pkl', 'wb'))
print("BM25 retriever serialized.")

