from langchain_ollama.chat_models import ChatOllama

# chat_model = ChatOllama(model="deepseek-r1:latest", reasoning=True)
chat_model = ChatOllama(model="qwen3:0.6b", reasoning=True)

result = chat_model.invoke("how many odd integers are greater than 0 and less than 10? Please be concise and state the final answer in JSON")

print(result)