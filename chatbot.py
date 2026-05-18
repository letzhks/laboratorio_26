# Il mio chatbot online

import streamlit as st
import pdfplumber

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.header("Assistenza online")

st.image("RAG classe/Chatbot.webp", width=400)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #12e08a;
        color: #B1D4D8;
    }
    </style>
    """,
    unsafe_allow_html=True)

with st.sidebar:
    st.title("Il mio documento")
    documento = st.file_uploader("Carica il tuo pdf:", type=["pdf"])

if documento is not None:
    with pdfplumber.open(documento) as pdf:
        # st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
        testo = ""
        for pagina in pdf.pages:
            testo = testo + pagina.extract_text() + "\n"
            # testo += pagina.extract_text() + "\n"
    # st.write(testo)

    taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
    
    frammenti = taglierina.split_text(testo)
    # st.write(f"Totale frammenti creati: {len(frammenti)}")
    # st.write(frammenti)

    # Generiamo gli embeddings
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    # st.write("Embedding recuperati!")

    # Salviamo gli embeddings in un vector store o vector db (es. FAISS, Pinecone, etc.)
    vettori = FAISS.from_texts(frammenti, embedding=embeddings)

    # Richiesta utente
    domanda_utente = st.text_input("Fai una domanda sul documento caricato:")

    # Generazione della risposta in una chain di eventi
    # domanda -> embedding -> similarity search -> risultati all'LLM -> risposta

    def formatta_documento(documenti):
        return "\n\n".join([documento.page_content for documento in documenti])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         '''Sei un assistente virtuale. 
    Usa il contesto fornito per rispondere alla domanda in modo conciso. 
    Non accedere a informazioni esterne, come Internet. 
    Se non conosci la risposta, dì semplicemente 'Non lo so'. 
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])
    
    comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})
    
    modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    catena = (
         {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
    
    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)
    



