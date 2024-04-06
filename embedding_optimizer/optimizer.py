import os
from openai import OpenAI
from langchain.docstore.document import Document

class EmbeddingOptimizer:

    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo-0125"):
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.client = OpenAI()
        self.model_name = model_name

    def model_summary(self, message):
        """
        Generate a summary based on the input message using the pre-defined model.

        This function sends a message to the model and retrieves a generated response.
        It is part of a class that has access to a client capable of communicating with an AI model.

        Args:
            message (str): The input message to summarize.

        Returns:
            str: The content of the first message choice from the model's response.
        """
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = [message]
        )
        return response.choices[0].message.content

    def summarize_each_part_independently(self, text, chunk_size=1000):
        """
        Summarizes each part of the input text independently and combines them.

        This function divides the input text into chunks of a specified size and
        uses a model to generate a summary for each chunk. The individual summaries
        are then combined into a final summary.

        Args:
            text (str): The input text to be summarized.
            chunk_size (int, optional): The size of text chunks that the input text is divided into.
                                        Defaults to 1000 characters.

        Returns:
            str: A combined summary of all the individual chunk summaries.
        """
        
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        subsummaries = []
        for chunk in text_chunks:
            message = {
            "role": "user", 
            "content": f"summarize this text: {chunk}"
            }
            subsummary = self.model_summary(message)
            subsummaries.append(subsummary)

        summary = '\n'.join(subsummaries)

        return summary

    def summarize_text_incrementally(self, text, chunk_size=1000):
        """
        Incrementally summarizes the text by processing it in chunks.

        This function breaks down the input text into chunks and incrementally builds a summary.
        Each chunk is summarized considering the context of the summary generated so far. The
        function keeps track of the last 3000 tokens of the current summary to provide context
        for summarizing the next chunk.

        Args:
            text (str): The full text to be summarized.
            chunk_size (int, optional): The size of each text chunk to be summarized, with a default of 1000 characters.

        Returns:
            str: The final summary, constructed incrementally from the individual chunk summaries.
        """
        
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        summary = ''
        for chunk in text_chunks:
            prompt = f'''
                You are currently writing the summary of a text. 
                Here you have the last 1000 tokens of your summary: {summary[-3000:]}
                Summarize this chunk so it can be added to your summary: {chunk}
            '''
            message = {
                "role": "user", 
                "content": prompt,
            }
            summary += self.model_summary(message)
        
        return summary

    def optimized_documents_for_accuracy(self, whole_document, chunk_documents):
        """
        Optimizes documents for accuracy by summarizing the whole document and appending it to a list.

        This function takes a whole document, summarizes it incrementally for better accuracy, and then
        creates a new Document object with the summary text. The new Document is then appended to a list
        of chunk documents.

        Args:
            whole_document (str): The complete document to be summarized.
            chunk_documents (list): A list of Document objects to which the new summary Document will be appended.

        Returns:
            list: The updated list of Document objects including the newly added summary Document.
        """
        
        summary_text = self.summarize_text_incrementally(whole_document)

        doc =  Document(page_content=summary_text, metadata={"source": "summary"})

        chunk_documents.append(doc)

        return chunk_documents

    def optimized_documents_for_storage(self, whole_document, chunk_documents):
        """
        Optimizes documents for storage efficiency by summarizing their content.

        This function iterates over a list of document objects, summarizing the content of each
        for more efficient storage. It also summarizes the entire provided document and appends
        this summary as a new document object to the list.

        Args:
            whole_document (str): The complete text of the document to be summarized.
            chunk_documents (list of Document objects): The list of document objects to be optimized.

        Returns:
            list of Document objects: The updated list with optimized content for each document and
                                    the summary of the whole document appended as a new Document object.
        """
        
        for document in chunk_documents:
            document.page_content = self.summarize_text_incrementally(document.page_content)
        
        summary_text = self.summarize_text_incrementally(whole_document)

        doc =  Document(page_content=summary_text, metadata={"source": "summary"})

        chunk_documents.append(doc)

        return chunk_documents