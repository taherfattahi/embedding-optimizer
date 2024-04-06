# Strategies for Efficient Data Embedding
Two approaches to generating optimized embeddings.

1) Creating Embeddings Optimized for Accuracy
  If you’re optimizing for accuracy, a good practice is to first summarize the entire document, then store the summary text and the embedding together. For the rest of the document, you can simply create overlapping chunks and store the embedding and the chunk text together.

![EmbeddingOptimizerAccuracy](https://raw.github.com/taherfattahi/embedding-optimizer/master/images/optimize-accuracy.webp
)

2) Creating Embeddings Optimized for Storage
  If you’re optimizing for space, you can chunk the data, summarize each chunk, concatenate all the summarizations, then create an embedding for the final summary.

![EmbeddingOptimizerStorage](https://raw.github.com/taherfattahi/embedding-optimizer/master/images/optimize-storage.webp
)


## Example
```python
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

```
### Installation

```sh
$ pip install embedding-optimizer
```
Also can be found on [pypi](https://pypi.org/project/embedding-optimizer)

### How can I use it?
  - Install the package by pip package manager.
  - After installing, you can use it and call the library.
  - 

### Issues
Feel free to submit issues and enhancement requests.

### Contributing
Please refer to each project's style and contribution guidelines for submitting patches and additions. In general, we follow the "fork-and-pull" Git workflow.

 1. **Fork** the repo on GitHub
 2. **Clone** the project to your own machine
 3. **Update the Version** inside __init__.py
 4. **Commit** changes to your own branch
 5. **Push** your work back up to your fork
 6. Submit a **Pull request** so that we can review your changes