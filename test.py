__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import time
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("openai_api_key")

parser = argparse.ArgumentParser()
parser.add_argument('-model',    help=' : model name', default='qwen2:7b')
parser.add_argument('-platform', help=' : platform name', default='ollama')
args = parser.parse_args()

from langchain.chat_models import ChatOpenAI

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

print("PLATFORM : " + args.platform)
print("MODEL    : " + args.model)

if args.platform == "ollama":
    model = ChatOllama(model=args.model)
elif args.platform == "huggingface":
    model = HuggingFacePipeline.from_model_id(
        model_id=args.model, task="text-generation", pipeline_kwargs={"max_new_tokens": 512},)
elif args.platform == "openai":
    if args.model == "openai":
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    elif args.model == "gpt-3.5":
        model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    elif args.model == "gpt-4":
        model = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    elif args.model == "gpt-4o":
        model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    else:
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
else:
    model = ChatOllama(model=args.model)

#response = model.invoke("앞으로 한글로 대답해주세요. 겨울철에 내한성이 강한 나무에는 어떤 것이 있을까요?")
#print(response)

loader = PyPDFLoader("a.pdf", extract_images=False)
pages = loader.load()

# 문장 임베딩 및 벡터 저장소 생성

# 문서를 문장으로 분리
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(pages)

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

import os
start = time.time()

if os.path.isdir("chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
end = time.time()
print(f"CREATE VECTOR DATABASE : {end - start:.5f} sec")

# 검색 쿼리
#query = "겨울철에 내한성이 강한 나무에는 어떤 것이 있을까요?"
# 가장 유사도가 높은 문장을 하나만 추출

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

#docs = retriever.get_relevant_documents(query)

# Prompt
#template = '''Answer the question based only on the following context:
#{context}
#
#Question: {question}
#'''

template = '''다음 내용만으로 대답을 해주시고 한글로 말해주세요:
{context}

질문: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("--------------------")
# Chain 실행
start = time.time()
query = "겨울철에 내한성이 강한 나무에는 어떤 것이 있을까요?"
answer = rag_chain.invoke(query)    
end = time.time()
print("Q : " + query)
print("A : " + answer)
print(f"LLM : {end - start:.5f} sec")

print("--------------------")
start = time.time()
query = "겨울철에 추위에 약한 나무에는 어떤 것이 있을까요?"
answer = rag_chain.invoke(query)
end = time.time()
print("Q : " + query)
print("A : " + answer)
print(f"LLM : {end - start:.5f} sec")

print("--------------------")
start = time.time()
query = "겨울철에 추위에 약한 나무 중에 이름이 제일 긴 나무 3개만 설명없이 이름만 출력하세요."
answer = rag_chain.invoke(query)
end = time.time()
print("Q : " + query)
print("A : " + answer)
print(f"LLM : {end - start:.5f} sec")
