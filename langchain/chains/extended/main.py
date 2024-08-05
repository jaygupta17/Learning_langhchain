from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableSequence
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt_template=ChatPromptTemplate.from_messages([
    ('system', "{smsg}"),
    ('human',"{hmsg}")
])

uppercase = RunnableLambda(lambda x : x.upper())
chain = prompt_template | llm | StrOutputParser() | uppercase

result=chain.invoke({'smsg':"You are Albert Einstien" , 'hmsg' : 'who are you?'})
print(result)