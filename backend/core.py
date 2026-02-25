from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.4
)

def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {'messages': [response]}

graph  = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)





    # thread_id = '1'
    # config = {'configurable', {'thread_id': thread_id}}
    #
    # while True:
    #     user_message = input("Type here: ")
    #
    #     print("User:",user_message)
    #
    #     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
    #         break
    #
    #     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]},config=config)
    #
    #     print('AI:',response['messages'][-1].content)

