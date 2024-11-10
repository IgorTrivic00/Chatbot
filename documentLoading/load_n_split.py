from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Putanja do PDF fajla
FILE_PATH = "../documents/bogati-otac-siromasni-otac.pdf"

# Učitavanje PDF-a
loader = PyPDFLoader(FILE_PATH)

# Splitovanje dokumenta na stranice
pages = loader.load_and_split()


# Kreiranje embedding funkcije
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Kreiranje vektorske baze podataka i perzistiranje
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_function,
    collection_name="bogati_otac_siromasni_otac",
    persist_directory="../vector_db"
)

# Perzistiranje vektorske baze podataka
vectordb.persist()