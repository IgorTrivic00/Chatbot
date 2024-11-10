import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from decouple import config
import os


# Onemogućavanje paralelizma za tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXT = [
    "Prvi svetski rat zapocet je atentatom na Austrougarskog cara Franca Ferinanda 1914. godine.",
    "Rat je trajao 4.godine i poginulo je preko 20 miliona ljudi",
    "Srbija je izgubila gotovo trecinu stanovnista u ratu.",
    "Srbija je imala velike pobede tokom rata u kolubarskoj i cerskoj bici.",
    "Rat je zavrsen probijanjem solunskog fronta u drugoj polovini 1918. godine."
]

meta_data = [
    {"source": "document 1", "page": 1},
    {"source": "document 2", "page": 2},
    {"source": "document 3", "page": 3},
    {"source": "document 4", "page": 4}
]

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever()
)

# Korišćenje invoke metode umesto get_relevant_documents
compressed_docs = compression_retriever.invoke("Kad je poceo i kad se zavrsio prvi svetski rat?")

print(compressed_docs)