from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda  , RunnableBranch
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


positive_feedback = ChatPromptTemplate.from_messages(
[
    ("system", "You are an helpful assistant who manages user feedback"),
    ("human", "reply the positive feedback of user:  {feedback}"),
]
)

negative_feedback = ChatPromptTemplate.from_messages(
[
    ("system", "You are an helpful assistant who manages user feedback"),
    ("human", "reply the negative feedback of user:  {feedback}"),
]
)

escalate_feedback = ChatPromptTemplate.from_messages(
[
    ("system", "You are an helpful assistant"),
    ("human", "Generate a message to Escalate the feedback to human agent:  {feedback}"),
]
)

positive_chain = positive_feedback | llm | StrOutputParser()
negative_chain = negative_feedback | llm | StrOutputParser()
escalate_chain = escalate_feedback | llm | StrOutputParser()

prompt = input("Enter feedback:")

prompt_temp = ChatPromptTemplate.from_messages([
    ('system' , 'You are an expert feedback reviewer'),
    ('human' , 'what is the category of given feedback among negative , positive , neutral ,escalate??"  : {feedback}')
])

branches = RunnableBranch(
    (
        lambda x: "POSITIVE" in x.upper(),
        positive_chain
    ),
    (
        lambda x: "NEGATIVE" in x.upper(),
        negative_chain
    ),
    escalate_chain
)

chain = prompt_temp | llm | StrOutputParser() | branches

print(chain.invoke({'feedback' : prompt}))

