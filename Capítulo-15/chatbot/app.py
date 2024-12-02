import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

with st.sidebar:
    st.title('🤗💬 CHATBOT')      
    st.write('Soy un chatbot analizador de PDFs')

import os
os.environ["OPENAI_API_KEY"] = "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"

def obtener_texto_pds(pdf_docs):
    texto = ""
    for pdf in pdf_docs:
        lector = PdfReader(pdf)
        for pagina in lector.pages:
            texto += pagina.extract_text()
    return texto
    
def obtenga_porciones_texto(texto):
    separador = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    porciones = separador.split_text(texto)
    return porciones
    
def obtener_vectores(porciones):
    embeddings = OpenAIEmbeddings()
    vectores = FAISS.from_texts(texts=porciones, embedding=embeddings) 
    return vectores

def obtener_salida(pregunta):
     try:
       if st.session_state.conversation is None:
          st.warning("Por favor, procesa un archivo antes de hacer preguntas.")
       else:
           respuesta = st.session_state.conversation({'question': pregunta})       
           st.session_state.chat_history = respuesta['chat_history']

           for i, mensaje in enumerate(st.session_state.chat_history):
               if i % 2 == 0:
                   st.write("️🤵Usuario:", mensaje.content)
               else:
                   st.write("🤖Bot:", mensaje.content)
     
     except Exception as e:
      print("Se ha producido una excepción: ", e)
       

def main():
    st.header("Ejemplo CHATBOT")
    st.subheader("Documentos")
   
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # cargue del pdf
    pdf = st.file_uploader("Cargue del PDF", type='pdf', accept_multiple_files=True)
                   
    if st.button("Procesar archivo"):
            with st.spinner("Procesando"):
                                                                                     
                # obtengo el texto de los pdfs
                texto_origen = obtener_texto_pds(pdf)
                #obtengo porciones del texto
                porciones_texto = obtenga_porciones_texto(texto_origen)
                #obtengo vectores
                vectores = obtener_vectores(porciones_texto)
                                                                                  
                llm = ChatOpenAI(temperature=0.5, max_tokens=1000, \
                      model_name="gpt-3.5-turbo")    
                                
                memoria = ConversationBufferMemory(memory_key='chat_history', \
                      return_messages=True)
                conversacion = ConversationalRetrievalChain.from_llm(
                     llm=llm,retriever=vectores.as_retriever(),memory=memoria
                     )
                
                st.session_state.conversation = conversacion     
                                                      
    consulta = st.text_input("Escribe una pregunta:")
    st.write(consulta)
                
    if consulta:
        obtener_salida(consulta)   
 
if __name__ == '__main__':
    main()