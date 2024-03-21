from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
#import openai
import os
import time

load_dotenv()

os.listdir(".")

# ファイルをアップロード

# loader = PyPDFLoader('日本財政の現状と望ましい税制の考察.pdf')
loader = PyPDFLoader("一般職業紹介状況.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

start = time.perf_counter()
# vectordb = Chroma.from_documents(texts, embeddings)
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="db/")
end = time.perf_counter()
print(end-start)

# qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectordb.as_retriever())
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 1}))

# プロンプトの定義
from langchain_core.prompts import PromptTemplate

template = """
あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
質問：{question}
回答：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

import gradio as gr

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    query = history[-1][0]

    query = prompt.format(question=query)
    #answer = qa.run(query)
    answer = qa.invoke(query)
    #print(type(answer))
    # source = qa._get_docs(query)[0]
    # source_sentence = source.page_content
    # answer_source = source_sentence +"\n"+"source:"+source.metadata["source"] + ", page:" + str(source.metadata["page"])
    # history[-1][1] = answer # + "\n\n情報ソースは以下です：\n" + answer_source
    history[-1][1] = answer # + "\n\n情報ソースは以下です：\n"
    return history

with gr.Blocks() as demo:
    #chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
    chatbot = gr.Chatbot([], elem_id="chatbot")

    with gr.Row():
        with gr.Column(scale=0.6):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
                container=False,
            )
            #).style(container=False)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

demo.launch()
