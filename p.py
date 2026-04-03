import streamlit as st
import os
from git import Repo

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

 
st.set_page_config(page_title="Codebase Explainer", layout="wide")
st.title("GitHub Codebase Explainer ")

INDEX_PATH = "faiss_index"
REPO_PATH = "repo"



@st.cache_resource
def load_or_create(repo_url):

    # embeddings + llm
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)

    # 🔥 Load existing index
    if os.path.exists(INDEX_PATH):
        vector_store = FAISS.load_local(INDEX_PATH, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return vector_store, retriever, llm

    # 🔥 Clone repo
    if not os.path.exists(REPO_PATH):
        Repo.clone_from(repo_url, REPO_PATH)

    documents = []

    for root, dirs, files in os.walk(REPO_PATH):
        if ".git" in root or "node_modules" in root:
            continue

        for file in files:
            if file.endswith((".py", ".js", ".java", ".cpp")):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        documents.append(
                            Document(
                                page_content=f.read(),
                                metadata={"file": file.lower()}
                            )
                        )
                except:
                    pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

  
    vector_store = FAISS.from_documents(chunks, embeddings)


    vector_store.save_local(INDEX_PATH)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    return vector_store, retriever, llm




repo_url = st.text_input("Enter GitHub Repo URL")

if st.button("Load Repo"):
    if not repo_url:
        st.error("Please enter a valid GitHub URL")
    else:
        with st.spinner("Processing repo..."):
            vector_store, retriever, llm = load_or_create(repo_url)

            prompt = PromptTemplate(
                template="""
You are a senior developer.

Rules:
- Answer ONLY from context
- If not found say "I don't know"
- Mention file name if possible

Context:
{context}

Question:
{question}
""",
                input_variables=["context", "question"]
            )

            def format_docs(docs):
                return "\n\n".join(
                    f"[File: {doc.metadata.get('file')}]\n{doc.page_content}"
                    for doc in docs
                )

            def smart_retrieval(query):
                results = [
                    doc for doc in vector_store.docstore._dict.values()
                    if query.lower() in doc.metadata.get("file", "")
                ]
                if results:
                    return results[:3]

                return retriever.invoke(query)

            parallel_chain = RunnableParallel({
                "context": RunnableLambda(smart_retrieval) | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            chain = parallel_chain | prompt | llm | parser

            st.session_state.chain = chain
            st.success("✅ Repo Ready!")



if "chain" in st.session_state:

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask about the code...")

    if user_input:
        st.session_state.history.append(("user", user_input))

        response = st.session_state.chain.invoke(user_input)
        st.session_state.history.append(("ai", response))


    for role, msg in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)