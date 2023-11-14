from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import spacy

from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS
len(STOP_WORDS)
nlp = spacy.load("en_core_web_sm")

def get_pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def preprocess(text):
  doc = nlp(text)
  no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
  return " ".join(no_stop_words)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len
        )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        llm=llm
        )
    return conversation_chain

def handle_message(conversation_chain, message):
    response = conversation_chain({'question': message})
    return response

def main():
    load_dotenv()
    pdf_docs = ['D:\Graduation project\lecture_1.pdf']
    #get pdf text
    row_text = get_pdf_text(pdf_docs)
    # print(f"{row_text}\nLength: {len(row_text)}")
    
    # #clean text
    clean_text = preprocess(row_text)
    # print(f"{clean_text}\nLength: {len(clean_text)}")

    #get pdf chunks
    text_chunks = get_text_chunks(clean_text)
    #create vector store
    vectorstore = get_vectorstore(text_chunks)
    #Create conversation chain
    conversation_chain = get_conversation_chain(vectorstore)
    #Handle message
    message = input("Enter your message: ")
    while(message != 'exit'):
        response = handle_message(conversation_chain, message)
        message = input("Enter your message: ")
        print(response)

if __name__ == '__main__':
    main()