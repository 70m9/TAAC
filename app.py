import chainlit as cl

persist_directory = Constants.PERSIST_DIRECTORY
import Constants
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import os



@cl.step(type="directoryExist")
async def directoryExist():
    # If the directory exists, ask the user before removing
    if os.path.exists(persist_directory):
        res = await cl.AskActionMessage(
            content="The directory '{persist_directory}' already exists. Do you want to overwrite it?",
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

@cl.step(type="createDatabase")
async def createDatabase():
    # If the directory exists, ask the user before removing
    tool_res = await directoryExist()
    
    # Create the directory if it doesn't exist
    tool_res = await directoryCreate()
    
    # Load Documents
    loader = DirectoryLoader("../Data", glob="*.pdf", loader_cls=PyPDFLoader)
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
            content="Database was created",
        ).send()
    else:
        await cl.Message(
            content="Operation cancelled. Exiting.",
        ).send()
        #exit()
    
    
        


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    
    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()