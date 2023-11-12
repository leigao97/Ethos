import openai
from openai import OpenAI

import json
from tqdm import tqdm
from datasets import load_from_disk

dataset = load_from_disk("./civil_comments_nontoxic")

prompt = "Based on the output provided below answered by LLM, predict the most likely input or instruction. Provide only the exact instruction without any additional explanation. \n Output: {toxic_output} \n Instruction:"
dictionary = {
    "instruction": "",
    "input": "",
    "output": ""
}

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)


message = [{"role": "system", "content": "You are a helpful assistant."}]
for i, sample in enumerate(dataset):
    print(sample["text"])
    message.append({"role": "user", "content": prompt.format(toxic_output=sample["text"])})
    if i > 10:
        break

response = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=message,
)
    # dictionary["instruction"] = response.choices[0].message.content
    # dictionary["output"] = sample["text"]
    # print
    # list.append(dictionary)
    # if i > 10:
    #     break
print(response)
# json_object = json.dumps(list, indent=4)

# with open("sample.json", "w") as outfile:
#     outfile.write(json_object)

# list = []
# for d in tqdm(dataset):
#     dictionary = {
#         "instruction": "",
#         "input": "",
#         "output": ""
#     }
#     dictionary["instruction"] = prompt.format(toxic_output=d["text"])
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#     )
#     dictionary["output"] = response['choices'][0]['text'].strip() 

#     list.append(dictionary)
#     break

# json_object = json.dumps(list, indent=4)

# with open("sample.json", "w") as outfile:
#     outfile.write(json_object)