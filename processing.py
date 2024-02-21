
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
    You are a native Londoner who is translating UK London slang into general English. 
    I will share a UK slang word or sentence with you and you will translate it to general English. 
    Your reply should be based on the UK slang word meanings, 
    and you will follow ALL of the rules below:

    1/ Response should be very similar or even identical to the UK slang word meanings, 
    in terms of words meaning, word choice, and word order.

    2/ If the UK slang word meanings are irrelevant, then try to mimic the style of the UK slang word meanings to prospect's message

    Below is a UK slang I need to translate to general English:
    {message}

    Here is a list of UK slang word meanings to help you with your response:
    {best_practice}

    Please translate the slang to generic English and output translation only:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=5)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    print(response)
    return response