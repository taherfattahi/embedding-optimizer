import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from embedding_optimizer.optimizer import EmbeddingOptimizer

# Set your OpenAI API Key
os.environ['OPENAI_API_KEY'] = ''

# Load your document
raw_document = TextLoader('test_data.txt').load()

# If your document is long, you might want to split it into chunks
text_splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_document)

embedding_optimizer = EmbeddingOptimizer(openai_api_key='')

# documents_optimizer = embedding_optimizer.optimized_documents_for_storage(raw_document[0].page_content, documents)
documents_optimizer = embedding_optimizer.optimized_documents_for_accuracy(raw_document[0].page_content, documents)

# Embed the document chunks and the summary
embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

db = FAISS.from_documents(documents_optimizer, embedding_model)

# query it
query = "What motivated Alex to create the Function of Everything (FoE)?"
docs = db.similarity_search(query)

print(docs[0].page_content)
