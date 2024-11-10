import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
#from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain

from decouple import config
import os
#import langchain.chains
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain.chains.conversational_retrieval import base as cr_base
#from langchain.chains.retrieval_qa import base as rq_base


#print(dir(langchain.chains))

#print(dir(langchain.chains.conversational_retrieval))
#print(dir(langchain.chains.retrieval_qa))
#print(dir(cr_base))
#print(dir(rq_base))

# Suzbijanje upozorenja za paralelizam i deprecacije
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="na_drini_cuprija",
    embedding_function=embedding_function,
)

# create prompt
QA_prompt = PromptTemplate(
    template="""Koristi sledeći kontekst da odgovoriš na pitanje na srpskom jeziku.
chat_history: {chat_history}
Context: {text}
Question: {question}
Answer:""",
    input_variables=["text", "question", "chat_history"]
)

# create chat model
llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0,  # Viša vrednost za kreativnije odgovore
    max_tokens=300)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    chain_type="refine",
)

# question
question = "Koji su glavni likovi u ovoj knjizi? Molim te navedi vise likova i lepo objasni."

# call QA chain
response = qa_chain.invoke({"question": question})

print(response.get("answer"))


