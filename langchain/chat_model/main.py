from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage , SystemMessage
from dotenv import load_dotenv 
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
chatHistory = [
    SystemMessage("")
]
while True:
    message = HumanMessage(input("You:"))
    chatHistory.append(message)
    ai_msg = llm.invoke(chatHistory)
    print(f"Ai : {ai_msg.content}")
    chatHistory.append(AIMessage(ai_msg.content))
