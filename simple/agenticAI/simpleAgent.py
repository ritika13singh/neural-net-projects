import requests


chat_history = [{"role":"system","content":"you are a helpful software developer assisstant"}]
def ask_ollama(prompt, model='llama3'):
   
    chat_history.append({"role":"user","content":prompt})

    url = 'http://localhost:11434/api/chat'

    payload = {
        'model': model,
        'messages': chat_history,
        'stream': False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        reply = response.json()["message"]["content"]
        chat_history.append({"role": "assistant", "content": reply})
        return reply
    else:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

# Example usage:
instruction_1 = input("Please enter you requirement:-")
response = ask_ollama(instruction_1)
instruction_2 = f"Parse the final version of python code from this and improve it by adding comprehensive documentation including: function description, parameter and return value description ${response}  "
response_2 = ask_ollama(instruction_2)


print(response)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,')
print(response_2)
