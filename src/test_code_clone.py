import json
import os
from tqdm import tqdm
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
        "temperature": 0,
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

def code_clone(code1, code2):
    question = "Please identify the following two codes is code clone or not, answer me just 'true' or 'false', no more other words\n"
    question += "code1: \n"
    question += code1
    question += "\n"
    question += "code2: \n"
    question += code2

    result = generate_llama_local(question)
    # print(result)
    predict = result["message"]["content"]
    predict = predict.lower()
    if predict == "true":
        return 1
    else:
        return 0

def load_big_code_clone():
    split = "test"
    json_file = os.path.join("../bc_data", "data.json")
    file = os.path.join("../bc_data", (split + ".txt"))

    codes_1 = []
    codes_2 = []
    labels = []
    json_data = {}

    with open(json_file, encoding='ISO-8859-1') as jf:
        lines = jf.readlines()
        print("loading dataset:")
        for line in tqdm(lines):
            data = json.loads(line.strip())
            source = data['code']
            json_data[data["idx"]] = {
                "code": source,

            }

    with open(file, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            try:
                ll = line.split("\t")
                if ll[0] not in json_data.keys() or ll[1] not in json_data.keys():
                    continue

                codes_1.append(json_data[ll[0]]["code"])
                codes_2.append(json_data[ll[0]]["code"])
                label = ll[2].replace("\n", "")
                labels.append(int(label))
            except Exception as e:
                # print(e)
                continue
    return codes_1, codes_2, labels


def compute_valid_metrics(predictions, labels):
    from sklearn.metrics import recall_score
    recall = recall_score(labels, predictions)
    from sklearn.metrics import precision_score
    precision = precision_score(labels, predictions)
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }
    print(result)

if __name__ == '__main__':
    codes_1, codes_2, labels = load_big_code_clone()
    predictions = []
    for code1, code2, label in tqdm(zip(codes_1, codes_2, labels)):
        predict=code_clone(code1, code2)
        predictions.append(predict)
    compute_valid_metrics(predictions, labels)



