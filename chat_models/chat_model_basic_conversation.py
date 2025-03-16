from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load the environment variables
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

result = model.invoke(messages)
print(f"Answer fron AI: {result.content}")

# AIMessage
messages = [
    SystemMessage("Solve the following math problems"),
    HumanMessage("What is 81 divided by 9?"),
    AIMessage("81 divided by 9 is 9"),
    HumanMessage("What is 9 times 9?"),
]

final_result = model.invoke(messages)
print(f"Final answer from AI: {final_result.content}")
