import streamlit as st
import logging
import os
import shutil
import re
import pdfplumber
from huggingface_hub import login
import json
import numpy as np
from langchain.chains import ConversationalRetrievalChain
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any
import requests
from huggingface_hub import InferenceClient
from bs4 import BeautifulSoup
import tempfile
import pgvector
import psycopg2
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

client = InferenceClient(
	provider="together",
	api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


llm_client = InferenceClient(
    model=model_id,
    timeout=120,
)


load_dotenv()


import psycopg2

st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

logger.info(f"data: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")

def get_vector_store(text_chunks):
    vector_store = None
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        logger.info("vector store is: ", vector_store)
    except Exception as e:
        logger.error(f"Error at get_vector_store function: {e}")
    return vector_store

@st.cache_resource
def get_embeddings():
    logger.info("Using Ollama Embeddings")
    return OllamaEmbeddings(model="nomic-embed-text")

def get_website_text(url: str) -> str:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = "\n".join([p.get_text() for p in soup.find_all("p")])
        return text
    except Exception as e:
        st.error(f"Error fetching website text: {e}")
        return ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        logger.info(chunks)
    except Exception as e:
        logger.error(f"Error at get_text_chunks function: {e}")
        chunks = []
    return chunks


@st.cache_resource
def get_llm():
    # logger.info(f"Using LLM mistral")

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_kwargs={"temperature":0.2})

    logger.info("llm is: ", llm)

    return llm
    

def line_list_output_parser(text: str) -> list[str]:
    logger.info("""Parses the output text into a list of lines.""")
    return text


def llm_model(question: str):
    return llm_client.text_generation(question)


def process_question(question: str, vector_db: FAISS) -> str:
    logger.info(f"Processing question: {question} {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")

    llm = get_llm()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
            Original question: {question}
        """,
    )

    retriever = vector_db.as_retriever()

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever, llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question as detailed as possible with reasoning from the provided context only. 
    Do not generate a factual answer if the information is not available, simply rely on data for answer. 
    If you do not know the answer, respond with "I don’t know the answer as not sufficient information is provided in the PDF."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Question processed and response generated: ", prompt)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)

    logger.info("Question processed and response generated: ", response)
    logger.info("Question processed and response generated: ", type(response))


    output = response.split("Answer:", 1)[-1].strip()

    logger.info("output is: ", output)

    if not response.strip():
        return "I don’t know the answer as not sufficient information is provided in the PDF."

    logger.info("Question processed and response generated: ", response)
    return output, response


def delete_vector_db() -> None:
    logger.info("Deleting vector DB")
    st.session_state.pop("pdf_pages", None)
    st.session_state.pop("file_upload", None)
    st.session_state.pop("vector_db", None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        logger.info("FAISS index deleted")
    st.success("Collection and temporary files deleted successfully.")
    logger.info("Vector DB and related session state cleared")
    st.rerun()

def main():
    st.subheader("Chat with PDF RAG", divider="gray", anchor=False)
    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "response_template" not in st.session_state:
        st.session_state["response_template"] = None

    pdf_docs = col1.file_uploader(
        "Upload your PDF Files", 
        accept_multiple_files=True
    )

    website_scraping = col1.text_input("Enter the url here 👇")

    col_buttons = col1.columns([1, 1])
    
    with col_buttons[0]:
        submit_button = st.button("Submit & Process", key="submit_process")

    with col_buttons[1]:
        delete_collection = st.button("⚠️ Delete collection", type="secondary")

    
    if submit_button and pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state["vector_db"] = get_vector_store(text_chunks)
            st.success("Done")

        pdf_pages = extract_all_pages_as_images(pdf_docs[0])
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider_1"
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    elif submit_button and website_scraping:
        with st.spinner("Processing..."):
            raw_text = get_website_text(website_scraping)
            text_chunks = get_text_chunks(raw_text)
            st.session_state["vector_db"] = get_vector_store(text_chunks)
            st.success("Done")


    if delete_collection:
        delete_vector_db()

    with col2:
        message_container = st.container(height=500, border=True)

        with message_container:
            for message in st.session_state["messages"]:
                avatar = "🤖" if message["role"] == "assistant" else "😎"
                with message_container.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
                try:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    message_container.chat_message("user", avatar="😎").markdown(prompt)

                    with message_container.chat_message("assistant", avatar="🤖"):
                        with st.spinner(":green[processing...]"):
                            if st.session_state["vector_db"] is not None:
                                output, response = process_question(
                                    prompt, st.session_state["vector_db"]
                                )
                                # print("Debug Response:", repr(response))
                                # print("messages are: ", message["content"])
                                st.markdown(output)
                                st.session_state["messages"].append(
                                    {"role": "assistant", "content": output}
                                )
                                st.session_state["response_template"] = response
                            if output:
                                with st.container():
                                    st.write()
                            else:
                                response = "Please upload and process a PDF file or website first."
                                st.warning(response)   
                except Exception as e:
                    st.error(e, icon="⚠️")
                    logger.error(f"Error processing prompt: {e}")

        # Ensure PDF viewer is retained
        if st.session_state.get("pdf_pages"):
            zoom_level = col1.slider(
                "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider_2"
            )
            with col1:
                with st.container(height=410, border=True):
                    for page_image in st.session_state["pdf_pages"]:
                        st.image(page_image, width=zoom_level)

    with col1:
        response_container = st.container(height=700, border=True)
        with response_container:
            if st.session_state.get("response_template"):
                st.subheader("🔍 Response Templates")
                st.write(f"**Extracted Response:** {st.session_state['response_template']}") 

if __name__ == "__main__":
    main()




# def process_question(question: str, vector_db: FAISS) -> str:
#     llm = get_llm()

#     context = retrieve_context(question, vector_db)

#     if not context.strip():
#         return "I don’t know the answer as not sufficient information is provided in the PDF."

#     # Generate response
#     response = llm.invoke(question)

#     # Validate response
#     if not validate_response(response, vector_db):
#         return "I don’t know the answer as not sufficient information is provided in the PDF."

#     return response






# prompt = "My favourite condiment is"

# model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
# model.to(device)

# generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
# tokenizer.batch_decode(generated_ids)[0]
# "My favourite condiment is to ..."


        # pdf_pages = extract_all_pages_as_images(pdf_docs[0])
        # st.session_state["pdf_pages"] = pdf_pages

        # zoom_level = col1.slider(
        #     "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider_1"
        # )

        # with col1:
        #     with st.container(height=410, border=True):
        #         for page_image in pdf_pages:
        #             st.image(page_image, width=zoom_level)




# def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
#     logger.info(f"Processing question: {question} using model: {selected_model}")
#     llm = get_llm()
    
#     template = """Answer the question as detailed as possible with reasoning from the provided context only. 
#     Do not generate a factual answer if the information is not available, simply rely on data for answer. 
#     If you do not know the answer, respond with "I don’t know the answer as not sufficient information is provided in the PDF."
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """

#     results = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 3})
#     # conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

#     logger.info("Question processed and response generated: ", results)

#     if results == 0 or results[0][1] < 0.7:
#         return "I don’t know the answer as not sufficient information is provided in the PDF."

#     logger.info("Question processed and response generated: ", results)
#     return response




    # response = llm_client.post(
    #     json={
    #         "inputs": question,
    #         "parameters": {"max_new_tokens": 200},
    #         "task": "text-generation",
    #     },
    # )

    # logger.info(json.loads(response.decode())[0]["generated_text"])

    # completion = client.chat.completions.create(
    #     model="mistralai/Mistral-Small-24B-Instruct-2501", 
    #     messages=question, 
    #     max_tokens=500,
    # )

    # logger.info(completion)

    # if not isinstance(question, str):
    #     question = str(question)

    # logger.info(type(question))


    # logger.info(f"Question passed to retriever: {question}, Type: {type(question)}")

    # retrieved_context = retriever.invoke(question)

    # logger.info(f"retrieved context is {retrieved_context}")

    # context_text = " ".join([doc.page_content for doc in retrieved_context])
    # logger.info(f"Context text: {context_text}")
    # logger.info(f"Retrieved context: {retrieved_context}")

    # outputs = llm.generate(vector_db.as_retriever(), max_new_tokens=20)

    # repo_id = "mistralai/Mistral-7B-v0.1"
    # llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv("HUGGING_FACE_SECRET"), 
    #                  repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})
    # return model
    # return ChatOllama(model="mistral", temperature=0.1)


    # @st.cache_resource
# def get_llm():
#     # logger.info(f"Using LLM mistral")

#     model_name = "google/flan-t5-large"
#     # tokenizer = T5Tokenizer.from_pretrained(model_name)
#     # model = T5ForConditionalGeneration.from_pretrained(model_name)

#     # model_name   = "google/flan-t5-large"
#     llm = pipeline("text2text-generation", model=model_name)
#     return llm
#     # repo_id = "mistralai/Mistral-7B-v0.1"
#     # llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv("HUGGING_FACE_SECRET"), 
#     #                  repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})
#     # return model
#     # return ChatOllama(model="mistral", temperature=0.1)


 
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    # model_name   = "google/flan-t5-large"


    
# DB_NAME=os.getenv("DB_NAME")
# USER=os.getenv("USER")
# PASSWORD=os.getenv("PASSWORD")
# HOST=os.getenv("HOST")
# PORT=os.getenv("PORT")



# @st.cache_resource(show_spinner=True)
# def extract_model_names() -> Tuple[str, ...]:
#     logger.info("Extracting model names")
#     models_info = ollama.list()
#     logger.info(f"Models Info Response: {models_info}")  # Debugging line

#     if "models" not in models_info or not isinstance(models_info["models"], list):
#         logger.error("Unexpected response structure from Ollama API")
#         return ()

#     model_names = tuple(model.get("name", "Unknown") for model in models_info["models"] if model.get("name") != "mistral")
#     logger.info(f"Extracted model names: {model_names}")
#     return model_names
