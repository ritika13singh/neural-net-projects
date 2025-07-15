from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")

chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "You are an AI assistant that helps users with travel plans. Where should I go in July for a beach vacation?"
response = chat(prompt, max_new_tokens=100, do_sample=True)
print(response[0]['generated_text'])
