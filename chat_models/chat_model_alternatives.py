from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load the environment variables
load_dotenv()

# Create a ChatOpenAI model
model_openai = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="2 + 2"),
]

# Invoke the model
result_openai = model_openai.invoke(messages)

print(f"OpenAI results: {result_openai.content}")

# Create a ChatGoogleGenerativeAI model
model_google = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Invoke the model
result_google = model_google.invoke(messages)

print(f"Google results: {result_google.content}")
