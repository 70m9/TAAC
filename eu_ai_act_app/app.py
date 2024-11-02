import chainlit as cl # type: ignore

import Constants
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
import shutil
import os

persist_directory = Constants.PERSIST_DIRECTORY

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = Constants.LANGCHAIN_API_KEY
    
HFHUB_API_KEY = Constants.HFHUB_API_KEY

from langchain import hub # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.llms import HuggingFaceHub # type: ignore
from langchain_chroma import Chroma # type: ignore

from langchain.chains import create_retrieval_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser # type: ignore
import re
from langchain.prompts import ChatPromptTemplate # type: ignore

from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore

from langchain.load import dumps, loads # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from operator import itemgetter
import textwrap



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

### Part 5.2 : Reciprocal Rank Fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

@cl.step(type="question_context")
async def question_context(result):
    pages_used(result)
    return
    
@cl.step(type="pages_used")
async def pages_used(result):
    context_documents = result["context"]
    # Create a list to store (page_number, page_content) tuples
    pages = []

        
    # Extract page number and content from each document
    for doc_tuple in context_documents:
        document, score = doc_tuple  # Unpacking the tuple
        page_number = document.metadata.get('page')
        source = document.metadata.get('source')
        page_content = document.page_content.strip()  # Clean up any whitespace
        
        # Append to the list as a tuple (page_number, page_content)
        pages.append((page_number, page_content))

    # Sort the pages by page number
    pages.sort(key=lambda x: x[0])  # Sort based on the first element of the tuple, which is the page number    
    
    for page_number, page_content in pages:
        message = f"Page {page_number}:"+"\n"
        message = message + page_content+"\n"
        message = message + "\n" + "-"*80 +"\n"
        
    await cl.Message(
        content="Pages used for Context: \n " + message,
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
            content=f"Operation denied.",
        ).send()
        #exit()
        
        
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    ## Part 3 : Generate Retreiver
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'tokenizer_kwargs': {'clean_up_tokenization_spaces': True}} # clean up extra spaces around special tokens
    )
    vectorstore = Chroma(persist_directory=Constants.PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    ## Part 4 : Initialize the LLM and prompt template
    # LLM
    # Corrected HuggingFaceEndpoint initialization
    llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-3B-Instruct", 
                        model_kwargs={"temperature": 0.4, "max_length": 200},
                        huggingfacehub_api_token=HFHUB_API_KEY)
    
    ## Part 5 :  RAG-Fusion
    ### Part 5.1 : Generate variations of the user question
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt
        | llm 
        | StrOutputParser() 
        |(lambda x: re.findall(r'\d+\.\s*(.+?)(?=\s*\d+\.|$)', x, re.MULTILINE)[-4:]) 
    )
    
    ### Part 5.2 : Reciprocal Rank Fusion
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    
    ### Part 5.2 : Invoke RAG-Fusion Chain
    # Improved template for question answering
    template = """You are a knowledgeable assistant tasked with answering questions based on provided context. 
    Please carefully read the context and provide a clear and accessible response that directly addresses the question. 

    **Context**:
    {context}

    **Question**: 
    {question}

    **Guidelines**:
    - Respond in a formal but accessible tone.
    - Summarize the purpose clearly and concisely, emphasizing the main goals without overly complex phrasing.
    - Avoid redundant details and focus on the essential points only.
    - Use at max 3 sentences.


    Provide your answer:
    """
    # Create the ChatPromptTemplate from the improved template
    prompt = ChatPromptTemplate.from_template(template)
    
    # LLM
    # Corrected HuggingFaceEndpoint initialization
    llm2 = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-3B-Instruct", 
                        model_kwargs={"temperature": 0.1, "max_length": 510},
                        huggingfacehub_api_token=HFHUB_API_KEY)
    
    # Prepare the final RAG chain with structured components
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt
        | llm2
        | StrOutputParser()
    )

    chain = RunnablePassthrough.assign(context=retrieval_chain_rag_fusion).assign(
        answer=final_rag_chain
    )

    keyword = "Provide your answer:"
    
    cl.user_session.set("chain", chain)
    cl.user_session.set("keyword", keyword)
    
    
        
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    keyword = cl.user_session.get("keyword")

    result = chain.invoke({"question": message.content})
    
    cl.user_session.set("result", result)
    
    
    # Send a response back to the user
    await cl.Message(
        content= "Generated Answer: \n" + textwrap.fill(result["answer"].split(keyword)[-1].strip(),width=125),
    ).send()
    
    ### Pages used for Context
    tool_res = await pages_used(result)

