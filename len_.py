import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


base_folder_path = "./"
in_folder_path = "/raid/hpc/hekai/WorkShop/My_project/Score_LLM/data/human_annotation_data_2_out_prompt.jsonl"
out_folder_path =  "_out_prompt_token.jsonl"


data = []
with open(in_folder_path,"r") as f:
    for line in f:
        obj = json.loads(line)
        data.append(obj)

data_token = []
for data1 in tqdm(data):
    data1_token = {}
    token_sum = 0
    for k in data1:
        input_ids = tokenizer.encode(data1[k], return_tensors='pt')
        token_length = len(input_ids[0])
        #print(token_length)
        token_sum += token_length
        data1_token[k] = token_length
    data1_token["sum"] = token_sum
    data_token.append(data1_token)
    #print(data_token)
    #break
import jsonlines
with jsonlines.open(out_folder_path, mode='w') as writer:
    for data1 in data_token:
        writer.write(data1)   