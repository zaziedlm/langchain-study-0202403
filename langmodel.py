from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9)
retmsg = llm.invoke("日本で一番高い山は何ですか？")
print(retmsg)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")
text = "日本で一番古い書物は何ですか？"
messages = [HumanMessage(content=text)]
retmsg = chat_model.invoke(messages)
print(retmsg)
