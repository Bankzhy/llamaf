from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True  # Optional: reduce memory usage
)

print("Model loaded successfully!")

prompt = """# Write a Python function to compute Fibonacci numbers:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.2,
    top_p=0.9,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
