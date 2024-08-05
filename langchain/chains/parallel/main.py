from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableParallel
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

prompt_temp = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product feature identifier"),
        ("human", "{product}"),
    ]
)

def get_product_pros(features):
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product feature reviewer"),
        ("human", "list pros of {features}"),
    ]
    )
    return prompt_template.format_prompt(features=features)

def get_product_cons(features):
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product feature reviewer"),
        ("human", "List cons of {features}"),
    ]
    )
    return prompt_template.format_prompt(features=features)

pros_branch = (RunnableLambda(lambda x: get_product_pros(x)) | llm | StrOutputParser())
cons_branch = (RunnableLambda(lambda x: get_product_cons(x)) | llm | StrOutputParser())

def combine_both(pros,cons):
    return f"Pros:\n{pros}\nCons:\n{cons}"

chain = (prompt_temp | llm | StrOutputParser() | RunnableParallel(branches={'pros':pros_branch , 'cons' : cons_branch}) | RunnableLambda(lambda x: combine_both(x['branches']['pros'] ,x['branches']['cons'])))

print(chain.invoke({'product':'macbook'}))

