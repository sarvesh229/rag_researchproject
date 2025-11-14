import os
import fitz
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# ----------------------------- #
# STEP 1: Load PDFs
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
# STEP 2: Create Vector Store
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
# STEP 3: Load Mistral Locally
# ----------------------------- #
def load_local_mistral_pipeline():
    print("🚀 Loading Mistral locally with 4-bit quantization...")
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.2
    )

    return HuggingFacePipeline(pipeline=pipe)

# ----------------------------- #
# STEP 4: Setup RetrievalQA
# ----------------------------- #
def setup_rag_pipeline(path, local_llm):
    embedding = HuggingFaceEmbeddings(
        model_name="deepset/gbert-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, chain_type="stuff")

# ----------------------------- #
# STEP 5: Save to PDF
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
    load_dotenv()

    # USER INPUT
    customer_name = input("Enter customer name: ")
    customer_id = input("Enter customer ID: ")
    user_prompt = input("Describe your furniture requirements: ")

    # Load and embed documents
    pdf_dir = "offers"
    docs = load_offer_documents(pdf_dir)
    vectordb_path = create_vectorstore(docs)

    # Load Mistral + run pipeline
    local_llm = load_local_mistral_pipeline()
    qa_chain = setup_rag_pipeline(vectordb_path, local_llm)
    response = qa_chain.run(user_prompt)

    # Save the result
    output_file = f"offer_{customer_id}.pdf"
    create_offer_pdf(response, customer_name, customer_id, output_file)

    print(f"\n🎉 Done! Your personalized offer has been saved as: {output_file}")
