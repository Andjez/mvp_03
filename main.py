#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      jackp
#
# Created:     15-07-2023
# Copyright:   (c) jackp 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import streamlit as st
from io import BytesIO
import tempfile
import zipfile
import shutil
import base64
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.schema import Document
from streamlit_chat import message
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import date

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
st.info('Please Click clear all button and also refresh the page before making new conversations', icon="ℹ️")

@st.cache_resource
def instructor_embeddings():
    instructor_embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
    return instructor_embed


def ori_data(file_zip):
    with tempfile.TemporaryDirectory() as temp_dir_01:
        # Extract files from zip
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_01)
        db = FAISS.load_local(temp_dir_01, embed)
        st.success('Database succussfully created!', icon="✅")
    return db


def history_data(file_zip):
    with tempfile.TemporaryDirectory() as temp_dir_02:
        # Extract files from zip
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_02)
        db = FAISS.load_local(temp_dir_02, embed)
        st.success('Database succussfully created!', icon="✅")
    return db

def youtube_loader(yt_link):
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def ori_data_02(_data_zip):
    db = FAISS.from_documents(data_zip, embed)
    st.success('Database succussfully created!', icon="✅")
    return db

def history_search(_query):
    history = st.session_state.db_history.similarity_search(query)
    return history

def update_history_db(_query,_answer):
    text_01 = [Document(page_content=query, metadata=dict(page="Question"))]
    text_02 = [Document(page_content=answer, metadata=dict(page="Answer"))]
    database = FAISS.from_documents(text_01, embed)
    st.session_state.db_history.merge_from(database)
    database = FAISS.from_documents(text_02, embed)
    st.session_state.db_history.merge_from(database)
    return st.session_state.db_history

def last_3():
    docstore_list = list(st.session_state.db_history.docstore._dict.values())
    last_element = []
    if len(docstore_list) > 2:
        for i in range(len(docstore_list)-1, len(docstore_list)-4, -1):
            last_element.append(docstore_list[i])
    else:
        for i in range(len(docstore_list)):
            last_element.append(docstore_list[i])
    return last_element

embed = instructor_embeddings()

chat = ChatOpenAI(temperature=0.0)

template_format ="""

Information from the Database:
{database_data}

Information from Conversation History:
{history_data}

Last 3 Conversations:
{Conversation_1_to_3}

Query: {User_query}

Response:"""

import shutil

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

prompt_template = ChatPromptTemplate.from_template(template_format)
query = ""

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "bool" not in st.session_state:
    st.session_state["bool"] = True

if "db_history" not in st.session_state:
    st.session_state["db_history"] = None

if "new_db" not in st.session_state:
    st.session_state.new_db = None

if "data_01" not in st.session_state:
    st.session_state.data_01 = False

if "data_02" not in st.session_state:
    st.session_state.data_02 = False

if "yt" not in st.session_state:
    st.session_state.yt = ""

from langchain.prompts import PromptTemplate

prompt_rephrase = PromptTemplate(
    input_variables=["user_Query","user_history"],
    template="""You are a llm model which rephrase user query more suitable to
    retieve informations from database which runs based on semantic search method in faiss database
    and if user query is not a complete one or lagging proper information use history of last three user querys to make a complete and suitable query for databse retriever
    user query:{user_Query}
    history of last three user querys:{user_history}
    """,
)

from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm_01 = OpenAI(
          model_name="text-davinci-003", # default model
          temperature=0.5) #temperature dictates how whacky the output should be
llmchain = LLMChain(llm=llm_01, prompt=prompt_rephrase)



if not st.session_state.data_01 & st.session_state.data_02:
    d = st.checkbox('Use existing database')
    col1, col2 = st.columns(2)
    if d:
        with col1:
            data_zip = st.file_uploader("Choose a Document Database as .zip file", type="zip",key=1)
            if data_zip is not None:
                bytes_data_zip = data_zip.getvalue()
                bytesio_zip = BytesIO(bytes_data_zip)
                st.session_state.new_db = ori_data(bytesio_zip)
                st.session_state.data_01 = True

        with col2:
                st.session_state.data_02 = False
                history_zip = st.file_uploader("Choose a History Database as .zip file", type="zip",key=2)
                if history_zip is not None:
                    bytes_data_zip = history_zip.getvalue()
                    bytesio_zip = BytesIO(bytes_data_zip)
                    st.session_state.db_history = history_data(bytesio_zip)
                    st.session_state.data_02 = True

    else:
        st.session_state.db_history = FAISS.from_texts("a", embed)
        st.session_state.data_02 = True
        text_input_container = st.empty()
        text_input_container.empty()
        yt_link = text_input_container.text_input("Enter Youtube link here!")
        if yt_link != "":
            st.session_state.yt = yt_link
            data_zip = youtube_loader(yt_link)
            st.session_state.new_db = ori_data_02(data_zip)
            st.session_state.data_01 = True

def clear_all():
    for i in range(3):
        #st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.yt = ""
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["bool"] = True
        st.session_state["db_history"] = None
        st.session_state.new_db = None
        st.session_state.data_01 = False
        st.session_state.data_02 = False

if  st.session_state.data_01 and st.session_state.data_02:
    with st.form("my_form"):
        query = st.text_input("Enter your question here:")
        submitted = st.form_submit_button("Submit")
    if submitted and query:
        last_conv = last_3()
        alter = llmchain.run({"user_Query": query, "user_history": last_conv})
        print(alter)
        docs = st.session_state.new_db.similarity_search(alter,k=2)
        his_qu = st.session_state.db_history.similarity_search(query,k=3)
        #last_conv = last_3()
        gen_messages = prompt_template.format_messages(database_data=docs,history_data=his_qu,Conversation_1_to_3=last_conv,User_query=query)
        response = chat(gen_messages)
        answer = response.content
        st.session_state.past.append(query)
        st.session_state.generated.append(answer)
        history = update_history_db(query,answer)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i],avatar_style="adventurer",seed=122, key=str(f"a{i}"))
            message(st.session_state["past"][i], avatar_style="adventurer",seed=121, is_user=True, key=str(i+9999) + "_user")


    with st.sidebar:
        if st.button("Clear all"):
            clear_all()

        if st.session_state.yt:
            st.video(st.session_state.yt)
        name = st.text_input("provide database name")
        if name:
            if st.button("Generate Database"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save the faiss file into temp folder
                    st.session_state.new_db.save_local(os.path.join(temp_dir, "history"))

                    # Create a ZipFile object
                    with zipfile.ZipFile(temp_dir + '.zip', 'w') as zipf:
                        # Navigate to the 'history' subdirectory
                        history_dir = os.path.join(temp_dir, "history")

                        # Loop through each file in the directory
                        for filename in os.listdir(history_dir):
                            # If the file is a regular file and not a directory
                            if os.path.isfile(os.path.join(history_dir, filename)):
                                # Add the file to the zip
                                zipf.write(os.path.join(history_dir, filename), arcname=filename)

                    # Provide the zipped file for download
                    with open(temp_dir + '.zip', 'rb') as f:
                        bytes = f.read()

                        st.download_button(
                            label="Download Document Database",
                            data=bytes,
                            file_name=f'Database_{name}.zip',
                            mime='application/zip',
                        )
            st.markdown("____________________")
            if st.button("Generate history"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save the faiss file into temp folder
                    st.session_state.db_history.save_local(os.path.join(temp_dir, "history"))

                    # Create a ZipFile object
                    with zipfile.ZipFile(temp_dir + '.zip', 'w') as zipf:
                        # Navigate to the 'history' subdirectory
                        history_dir = os.path.join(temp_dir, "history")

                        # Loop through each file in the directory
                        for filename in os.listdir(history_dir):
                            # If the file is a regular file and not a directory
                            if os.path.isfile(os.path.join(history_dir, filename)):
                                # Add the file to the zip
                                zipf.write(os.path.join(history_dir, filename), arcname=filename)

                    # Provide the zipped file for download
                    with open(temp_dir + '.zip', 'rb') as f:
                        bytes = f.read()

                        st.download_button(
                            label="Download History Database",
                            data=bytes,
                            file_name=f'history_{name}_{date.today()}.zip',
                            mime='application/zip',
                        )

