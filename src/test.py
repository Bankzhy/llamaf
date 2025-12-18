import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def batch_classify(codes):
    prompts = [
        f"""Classify the following code as buggy or clean.
Only output the label.

Code:
{code}

Label:"""
        for code in codes
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=False,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.split("Label:")[-1].strip() for d in decoded]

if __name__ == '__main__':
    codes = [
        """
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr
        
        data = [64, 34, 25, 12, 22, 11, 90]
        sorted_data = bubble_sort(data)
        print(sorted_data)

        """
    ]

    result = batch_classify(codes)
    print(result)


