import ollama

msg = []
msg.append({"role": "user", "content": "O que é o Ollama?"})

response = ollama.chat(model="llama3:8b-instruct-q2_K", messages=msg)
print(response['message']['content'])