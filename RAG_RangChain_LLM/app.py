from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# 모든 모듈을 langchain에서 가져오기
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Step 1: Load the model and tokenizer
model_id = "kyujinpy/Ko-PlatYi-6B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=True,
    max_new_tokens=300,
)

# Step 2: Define the chat template and the chain
prompt_template = """
You are a helpful assistant. Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}

Provide a direct answer based on the context provided above:
"""

koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = LLMChain(llm=koplatyi_llm, prompt=prompt)

# Step 3: Load and process PDF with RAG
loader = PyPDFLoader("/home/LLM/intel.pdf")
pages = loader.load_and_split()

#청크사이즈 조절
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50) 
texts = text_splitter.split_documents(pages)

model_name = "jhgan/ko-sbert-nli"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={'normalize_embeddings': True}
)

db = FAISS.from_documents(texts, hf_embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    # 모델을 사용하여 응답 생성
    result = rag_chain.invoke(user_message)
    response_text = result["text"].strip()

    # 기본 응답 처리
    if "Answer:" in response_text:
        response_text = response_text.split("Answer:")[-1].strip()
    elif not response_text:
        response_text = "죄송합니다, 해당 날짜에 대한 정보를 찾을 수 없습니다."
    else:
        response_text = "공강 또는 주말이였습니다."

    # 응답 포맷팅
    response_text = f"{response_text}"

    response = {
        "message": response_text
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
