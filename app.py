import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(page_title="DocumentGPT", page_icon="📄")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_resource
def get_memory():
    return ConversationBufferMemory(return_messages=True)


memory = get_memory()


@st.cache_data(
    show_spinner="Embedding file...",
)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embedded_files/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()

    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vector_store = FAISS.from_documents(docs, cache_embeddings)

    retriver = vector_store.as_retriever()
    return retriver


def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message: str, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state.messages:
        send_message(
            message=message["message"],
            role=message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer questions using only the following context. If you don't know the answer
            just say you don't know. Don't make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions about the document you uploaded.

Upload your files on the sidebar and start asking questions.
"""
)


with st.sidebar:
    file = st.file_uploader(
        "Upload a file(.txt, .pdf, .docx)", type=["txt", "pdf", "docx"]
    )
if file:
    retriver = embed_file(file)

    send_message("I have successfully loaded the document", role="ai", save=False)
    paint_history()
    message = st.chat_input("Ask me anything about the document...")

    if message:
        send_message(message=message, role="human")
        chain = (
            {
                "context": retriver | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(load_memory),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
        memory.save_context(
            {"input": message},
            {"output": response.content},
        )

else:
    st.session_state.messages = []
    memory = get_memory()
