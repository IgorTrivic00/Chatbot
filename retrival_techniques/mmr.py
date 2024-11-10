from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

TEXT = ["Prvi svetski rat zapocet je atentatom na Austrougarskog cara Franca Ferinanda 1914. godine.",
        "Rat je trajao 4.godine i poginulo je preko 20 miliona ljudi",
        "Srbija je izgubila gotovo trecinu stanovnista u ratu."
        "Srbija je imala velike pobede tokom rata u kolubarskoj i cerskoj bici.",
        "Rat je zavrsen probijanjem solunskog fronta u drugoj polovini 1918. godine."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

response = vector_db.max_marginal_relevance_search(
    query="Kada je poceo i kada se zavrsio prvi svetski rat?", k=2)

print(response)