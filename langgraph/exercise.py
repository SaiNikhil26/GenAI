import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import json

# Load environment variables
load_dotenv()

# Define the state schema
class CodeAnalyzerState(TypedDict):
    query: str  # The input code to analyze
    messages: Annotated[List, add_messages]  # List of all messages
    bugs_fixed_count: int  # Counter for bugs fixed
    improvements_count: int  # Counter for improvements suggested
    last_bug_fix_suggestion: str  # Latest bug fix suggestion
    last_improvement_suggestion: str  # Latest improvement suggestion
    has_bugs: bool  # Flag to indicate if code has bugs

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Node 1: Determine if the code has bugs
def check_for_bugs(state: CodeAnalyzerState) -> CodeAnalyzerState:
    """
    Analyze the code to determine if it has bugs.
    """
    code = state["query"]

    prompt = f"""
    You are a Python code analyzer. Analyze the following code and determine if it has any bugs or syntax errors.
    Respond with 'Yes' if there are bugs or 'No' if there are no bugs. Don't provide any explanations.

    ```python
    {code}
    ```
    """

    response = model.invoke(prompt)
    has_bugs = "yes" in response.content.lower()

    # Add message to the state
    return {
        "messages": [HumanMessage(content=f"Checking for bugs in code:\n{code}"),
                    AIMessage(content=f"Bug check result: {'Yes' if has_bugs else 'No'}")],
        "has_bugs": has_bugs
    }

# Node 2: Fix bugs in the code
def fix_bugs(state: CodeAnalyzerState) -> CodeAnalyzerState:
    """
    Identify bugs in the code and suggest fixes.
    """
    code = state["query"]

    prompt = f"""
    You are a Python debugging expert. Analyze the following code and identify any bugs or syntax errors.
    Provide a detailed explanation of each issue found and suggest how to fix it.
    Do NOT provide the complete fixed code, only explain the issues and suggest solutions.

    ```python
    {code}
    ```
    """

    response = model.invoke(prompt)
    bug_fix_suggestion = response.content

    # Update the state
    return {
        "messages": [HumanMessage(content=f"Finding bugs in code:\n{code}"),
                    AIMessage(content=bug_fix_suggestion)],
        "bugs_fixed_count": state.get("bugs_fixed_count", 0) + 1,
        "last_bug_fix_suggestion": bug_fix_suggestion
    }

# Node 3: Suggest improvements for the code
def suggest_improvements(state: CodeAnalyzerState) -> CodeAnalyzerState:
    """
    Suggest improvements for the code that has no bugs.
    """
    code = state["query"]

    prompt = f"""
    You are a Python optimization expert. The following code is syntactically correct and has no bugs.
    Analyze it and suggest improvements for:
    1. Better performance
    2. Following Python best practices
    3. Code readability and maintainability
    4. Any other potential improvements

    Provide detailed explanations for each suggestion. Do NOT provide the complete improved code.

    ```python
    {code}
    ```
    """

    response = model.invoke(prompt)
    improvement_suggestion = response.content

    # Update the state
    return {
        "messages": [HumanMessage(content=f"Finding improvements for code:\n{code}"),
                    AIMessage(content=improvement_suggestion)],
        "improvements_count": state.get("improvements_count", 0) + 1,
        "last_improvement_suggestion": improvement_suggestion
    }

# Router function to decide the next node
def router(state: CodeAnalyzerState) -> str:
    """
    Route to the appropriate node based on whether the code has bugs.
    """
    if state.get("has_bugs", False):
        return "fix_bugs"
    else:
        return "suggest_improvements"

# Create the graph
def create_code_analyzer_graph():
    # Initialize the graph with our state schema
    workflow = StateGraph(CodeAnalyzerState)

    # Add nodes
    workflow.add_node("check_for_bugs", check_for_bugs)
    workflow.add_node("fix_bugs", fix_bugs)
    workflow.add_node("suggest_improvements", suggest_improvements)

    # Add edges
    #workflow.add_edge("check_for_bugs", router)
    workflow.add_conditional_edges(
        "check_for_bugs",
        router,
        {
            "fix_bugs": "fix_bugs",
            "suggest_improvements": "suggest_improvements"
        }
    )
    workflow.add_edge("fix_bugs", END)
    workflow.add_edge("suggest_improvements", END)

    # Set the entry point
    workflow.set_entry_point("check_for_bugs")

    # Create memory saver for state persistence
    memory = MemorySaver()

    # Compile the graph
    return workflow.compile(checkpointer=memory)

# Sample code with bugs
buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This will cause a division by zero error
result = calculate_average([])
print(f"The average is: {result}")
"""

# Sample code with no bugs but needs improvement
improvable_code = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

# Calculate first 10 Fibonacci numbers
result = fibonacci(10)
print(result)
"""

# Another sample with bugs
buggy_code_2 = """
def find_max(lst):
    if len(lst) == 0:
        return None
    max_val = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
    return max_val

# This will cause an index error
my_list = []
print(f"Maximum value: {find_max(my_list)}")
"""

# Another sample that needs improvement
improvable_code_2 = """
def search_element(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Search for an element in a list
my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
index = search_element(my_array, 7)
print(f"Element found at index: {index}")
"""

if __name__ == "__main__":
    # Create the graph
    app = create_code_analyzer_graph()

    # Test with buggy code
    print("\n===== ANALYZING CODE WITH BUGS =====")
    result1 = app.invoke({
        "query": buggy_code,

    }, {"configurable": {"thread_id": "1"}})

    print("\nstate after ANALYZING CODE WITH BUGS========")
    print(json.dumps(result1, indent=2, default=str))

    # Test with improvable code
    print("\n===== ANALYZING CODE THAT NEEDS IMPROVEMENT =====")
    result2 = app.invoke({
        "query": improvable_code,

    }, {"configurable": {"thread_id": "1"}})

    print("\nstate after ANALYZING CODE THAT NEEDS IMPROVEMENT========")
    print(json.dumps(result2, indent=2, default=str))



    # # You can add more test cases here
    # print("\n===== ANALYZING ANOTHER CODE WITH BUGS =====")
    # result3 = app.invoke({
    #     "query": buggy_code_2,
    #     "messages": [],
    #     "bugs_fixed_count": 0,
    #     "improvements_count": 0,
    #     "last_bug_fix_suggestion": "",
    #     "last_improvement_suggestion": "",
    #     "has_bugs": False
    # }, {"configurable": {"thread_id": "3"}})

    # print("\nBug Fix Suggestion:")
    # print(result3.get("last_bug_fix_suggestion", "No bug fix suggestions"))
