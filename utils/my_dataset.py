from torch.utils.data import Dataset
import json


class My_dataset(Dataset):

    def __init__(self, data_file):
        
        self.data = []
        with open(data_file, 'r') as file:
            for i in file:
                self.data.append(json.loads(i))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[index]
        question = data_item['question']
        answs = eval(data_item['possible_answers'])

        return {"question": question,  "answer": answs}