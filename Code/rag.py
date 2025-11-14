import os
import fitz
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import HuggingFaceHub

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ----------------------------- #
# STEP 1: Hugging Face Login
# ----------------------------- #
def huggingface_login():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("❌ HF_TOKEN not found in .env file.")
    login(token=hf_token)
    print("🔐 Hugging Face login successful.")

# ----------------------------- #
# STEP 2: Load & Split Offers
# ----------------------------- #
def load_offer_documents(folder_path):
    print(f"📂 Loading PDFs from '{folder_path}'...")
    all_docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(all_docs)

# ----------------------------- #
# STEP 3: Create FAISS Vector Store
# ----------------------------- #
def create_vectorstore(docs, path="faiss_index"):
    print("🧠 Creating FAISS vector store with gbert-large...")
    embedding = HuggingFaceEmbeddings(
        model_name="deepset/gbert-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = FAISS.from_documents(docs, embedding)
    vectordb.save_local(path)
    return path

# ----------------------------- #
# STEP 4: Setup RAG Chain
# ----------------------------- #
def setup_rag_pipeline(path):
    embedding = HuggingFaceEmbeddings(
        model_name="deepset/gbert-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        task="text-generation"  # ✅ This line resolves the Pydantic error
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# ----------------------------- #
# STEP 5: Generate Offer PDF
# ----------------------------- #
def create_offer_pdf(content, customer_name, customer_id, filename):
    today = datetime.now().strftime("%d.%m.%Y")
    header = f"""Offer for: {customer_name}\nCustomer ID: {customer_id}\nDate: {today}\n\n"""
    full_text = header + content

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), full_text, fontsize=11)
    doc.save(filename)
    doc.close()
    print(f"✅ PDF offer saved to: {filename}")

# ----------------------------- #
# MAIN EXECUTION
# ----------------------------- #
if __name__ == "__main__":
    huggingface_login()

    # 🧑 USER INPUT
    customer_name = input("Enter customer name: ")
    customer_id = input("Enter customer ID: ")
    user_prompt = input("Describe your furniture requirements: ")

    # 📂 Folder with offer PDFs
    pdf_dir = "offers"  # <-- Ensure this folder exists and contains your PDF offers

    # 🔁 Run the RAG Pipeline
    docs = load_offer_documents(pdf_dir)
    vectordb_path = create_vectorstore(docs)
    qa_chain = setup_rag_pipeline(vectordb_path)
    response = qa_chain.run(user_prompt)

    # 📄 Save to PDF
    output_file = f"offer_{customer_id}.pdf"

    create_offer_pdf(response, customer_name, customer_id, output_file)

    print(f"\n🎉 Done! Your personalized offer has been saved as: {output_file}")
