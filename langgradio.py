from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.invoke(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()