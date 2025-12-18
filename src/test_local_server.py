import requests

def generate_llama_local(content, history=[]):
    msg = history
    msg.append(
        {
            "role": "user",
            "content": content
        }
    )

    data = {
        "model": "codellama:7b-instruct",
        # "temperature": 0,
        "messages": msg,
        "stream": False,
        "options": {
        }
    }

    # data =  {
    #         "role": "user",
    #         "content": content
    # }

    response = requests.post("http://localhost:11434/api/chat", json=data)

    # api = APIMaster.objects.get(api_name="llama3")
    # api.api_current_count += len(ques)
    # api.save()

    result = response.json()
    # result = result["message"]["content"]
    # print(result)
    return result


def generate_ast(code):
    question = "Please generate the PDG based on the following code with json format\n"
    question += code

    result = generate_llama_local(question)
    print(result)

def code_clone(code1, code2):
    question = "Please identify the following two codes is code clone or not, answer 'true' or 'false'\n"
    question += "code1: \n"
    question += code1
    question += "\n"
    question += "code2: \n"
    question += code2

    result = generate_llama_local(question)
    print(result)

if __name__ == '__main__':
    code = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
    """

    code1 = """
    def sum_positive(nums):
        total = 0
        for n in nums:
            if n > 0:
                total += n
        return total
    """

    code2 = """
    def add_positive(numbers):
        result = 0
        for value in numbers:
            if value > 0:
                result = result + value
        return result
    """

    code_clone(code1, code2)