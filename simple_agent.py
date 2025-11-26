import os
import dotenv

dotenv.load_dotenv()

# 1. Building an LLM

from llama_index.llms.openai import OpenAI
llm1 = OpenAI(
    model="gpt-3.5-turbo-0613",
    api_key = os.getenv("OPENAI_API_KEY"),
    api_base = os.getenv("OPENAI_API_URL")
)

# simple print
# print(llm.complete("William Shakespeare is "))

# streaming print
# response = llm.stream_complete("William Shakespeare is ")
# for r in response:
#     print(r.delta, end="")
    
# async complete - acomplete(), async streaming - astream_complete()

# Chat interface
from llama_index.core.base.llms.types import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Tell me a joke")
]

# chat response
chat_response = llm1.chat(messages)

# stream chat response
stream_chat_response = llm1.stream_chat(messages)
for response in stream_chat_response:
    # print(response.delta, end="")
    pass

# for aync - astream_chat()


# Multi-Modal LLMs - LLama support text, audio, images inside ChatMessage using content blcok
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock

llm2 = OpenAI(
    model="gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
    api_base = os.getenv("OPENAI_API_URL")
)

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="./hurain.jpg"),
            TextBlock(text="Describe the image in a few sentence")
        ],
    )
]

response = llm2.chat(messages)
# print(response.message.content)

# Tool Calling - giving hands to LLM

from llama_index.core.tools import FunctionTool

def generate_song(name: str, artist: str) -> str:
    """Generate a song with provided name and artist"""
    return {"name": name, "artist": artist}


# tool1 = FunctionTool.from_defaults(fn=generate_song)

#response = llm2.predict_and_call(
    #tools = [tool1],
    #user_msg = "pick a song for me",   # calculate only multiplication. For multi tool call we have to manually wrap the things 
#)

# print(response)


# 2. Building an Agent - FunctionAgent, AgentWorkflow (capable of managing multi agent) and more
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow

# Creating basic tools
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result"""
    return a + b

tool1 = FunctionTool.from_defaults(fn=multiply)
tool2 = FunctionTool.from_defaults(fn=add)

tools = [tool1, tool2]
# initializing agent
# it requires a array of tools, llmss and system prompt (detailed)

workflow = FunctionAgent(
    tools=tools,
    llm = llm2, 
    system_prompt="You are an agent that can perform basic mathematical operations using provided tools."
)

# Note - Other agent also availabe other than FunctionAgent
# 1. ReActAgent - Reasoning and Action Agent
# 2. CodeActAgent and more

# asking question

# async def main():
#     response = await workflow.run(user_msg="What is 20+(2*4)?")
#     print(response)
    
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
    
    
# 3. Using existing tools
from llama_index.llms.openai import OpenAI

llm3 = OpenAI(
    model = "gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_URL")
)

from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

workflow2 = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations",
    llm = llm3,
    tools = finance_tools,
    system_prompt="You are a helpful financial assistant."
)

# async def main():
#     response = await workflow2.run(user_msg="What is the current stock price of NVIDIA")
#     print(response)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
    
    
# 6. Maintaining State
# Agents are stateless in general. To maintain state, we need to keep track of the previous state.
# using context class in llamaindex - maintaining state within and between runs

from llama_index.core.workflow import Context

# to maintain state between runs we create a new Context class ctx

ctx1 = Context(workflow2)

# Now we can pass it to our first run

# async def main():
#     response = await workflow2.run(user_msg="Hi! My name is Hasnain.", ctx=ctx1)
#     print(response)
#     response2 = await workflow2.run(user_msg="What is my name?", ctx=ctx1)
#     print(response2)
    
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
    
    
# Maintaining state over longer period
# The context is serializable, so it can be saved to DB, file etc. and loaded back in later

# JsonSerializer - user json.dumps() and json.loads() to serialize and deserialize the context
# JsonPickleSerializer - uses pickle to serialize and deserialize the context. If object in context is not serializable then we can use it

from llama_index.core.workflow import JsonSerializer, JsonPickleSerializer

# we can then serialize our context to dict and save it to a file

ctx_dict = ctx1.to_dict(serializer=JsonSerializer())
restored_ctx = Context.from_dict(
    workflow2, ctx_dict, serializer=JsonSerializer()
)

# async def main():
#     response3 = await workflow2.run(user_msg="What's my name?", ctx= restored_ctx)
#     print(response3)
    
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
    
    
# Tools and State
# Tools also have access to workflow context that means we can set and retrieve variables from the context and use them in the tools or to pass info between tools

# AgentWorkflow - used a context variable "state" 
# To access context - context parameter should be the first parameter of the tool

# Async tool
async def set_name(ctx: Context, name: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["name"] = name
    return f"Name set to {name}"

workflow4 = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm = llm2,
    system_prompt="You are a helpful assistant that can set name.",
    initial_state={"name": "unset"}
)

ctx2 = Context(workflow4)

async def main():
    response1 = await workflow4.run(user_msg ="What's my name?", ctx=ctx2)
    print(response1)
    
    response2 = await workflow4.run(user_msg="My name is Uzaif", ctx=ctx2)
    print(response2)
# Accessing the value of state directly
    state = await ctx2.store.get("state")
    print(state["name"])
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    

