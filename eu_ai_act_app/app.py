import chainlit as cl # type: ignore

import Constants
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
import shutil
import os

from langchain import hub # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.llms import HuggingFaceHub # type: ignore
from langchain_chroma import Chroma # type: ignore

persist_directory = Constants.PERSIST_DIRECTORY
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = Constants.LANGCHAIN_API_KEY
    
HFHUB_API_KEY = Constants.HFHUB_API_KEY


@cl.step(type="createDatabase")
async def createDatabase():
    
    # If the directory exists, ask the user before removing
    tool_res = await directoryExist()
    
    # Create the directory if it doesn't exist
    tool_res = await directoryCreate()
    
    # Load Documents
    loader = DirectoryLoader("./Data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
 
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'tokenizer_kwargs': {'clean_up_tokenization_spaces': True}}
    )

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    return "Created and persisted new database."


@cl.step(type="directoryExist")
async def directoryExist():
    # If the directory exists, ask the user before removing
    if os.path.exists(persist_directory):
        res = await cl.AskActionMessage(
            content="The directory "'{persist_directory}'" already exists. Do you want to overwrite it?",
            actions=[
                cl.Action(name="yes", value="yes", label="✅ Yes"),
                cl.Action(name="no", value="no", label="❌ No"),
            ],
        ).send()
        #   talvez precisse
        # await action.remove()
        
        if res and res.get("value") == "yes":
            shutil.rmtree(persist_directory)
            await cl.Message(
                content="Removing existing directory: {persist_directory}",
            ).send()
        else:
            await cl.Message(
                content="Operation cancelled. Exiting.",
            ).send()
            #exit()
    return

@cl.step(type="directoryCreate")
async def directoryCreate():
    # Create the directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    await cl.Message(
        content="Created directory: {persist_directory}",
    ).send()
    return


@cl.on_chat_start
async def main():
    res1 = await cl.AskActionMessage(
            content="Do you want to create a new database??",
            actions=[
                cl.Action(name="yes", value="yes", label="✅ Yes"),
                cl.Action(name="no", value="no", label="❌ No"),
            ],
        ).send()
        #   talvez precisse
        # await action.remove()
        
    if res1 and res1.get("value") == "yes":
        tool_res = await createDatabase()
        await cl.Message(
            content=f"Database was created",
        ).send()
    else:
        await cl.Message(
            content=f"Operation cancelled. Exiting.",
        ).send()
        #exit()
        
    # Generate Retreiver
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'tokenizer_kwargs': {'clean_up_tokenization_spaces': True}} # clean up extra spaces around special tokens
    )
    vectorstore = Chroma(persist_directory=Constants.PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    ## Part 4 : Initialize the LLM and prompt template
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # Corrected HuggingFaceEndpoint initialization
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", 
                        model_kwargs={"temperature": 0.5, "max_length": 512},
                        huggingfacehub_api_token=HFHUB_API_KEY)
    
    ## Part 5 : Define RAG Chain
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("rag_chain", rag_chain)
    
        


@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    result = rag_chain.invoke(message.content)
    
    # Send a response back to the user
    await cl.Message(
        content=f"{result}",
    ).send()
