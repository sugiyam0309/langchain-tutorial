from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

model = init_chat_model("gpt-3.5-turbo",
                        model_provider="openai",)

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)
print(response.content)

response = model.invoke("Hello")
print(response.content)

response = model.invoke([{"role": "user", "content": "Hello"}])
print(response.content)

response = model.invoke([HumanMessage("Hello")])
print(response.content)

for chunk in model.stream(messages):
    print(chunk.content, end="|", flush=True)


system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

response = model.invoke(prompt)
print(response.content)
