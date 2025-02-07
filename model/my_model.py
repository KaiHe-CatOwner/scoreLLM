# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from accelerate import Accelerator
from utils.utils import clip_path_map


class MyCustomModel(nn.Module):
    def __init__(self, llm_requires_grad, load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token, tokenizer):
        nn.Module.__init__(self)

        self.llm_tokenizer = tokenizer
        self.llm = self.load_llm(load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token)
       
        self.config = self.llm.config
        self.llm.requires_grad = llm_requires_grad


    
    def generate(self, *args, **kwargs):
        generation_config = GenerationConfig(
                max_length=kwargs["max_length"],
                max_new_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
                top_p=kwargs["top_p"],
                num_return_sequences=kwargs["num_return_sequences"],
                repetition_penalty=kwargs["repetition_penalty"],
                do_sample=kwargs["do_sample"],
                pad_token_id=kwargs["pad_token_id"],
                bos_token_id=kwargs["bos_token_id"],
            )
        
        with torch.no_grad():
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
          
            res = self.llm.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)

        generate_list = []
        for item in res:
            generation = self.llm_tokenizer.decode(item, skip_special_tokens=True)
            generate_list.append(generation)
        return generate_list


    def load_llm(self, load_in_8bit, load_in_4bit, llm_name, trust_remote_code, token):
        print("llm loading ...")
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
            )
            # Copy the model to each device
            device_map = {"": Accelerator().local_process_index}
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None


        llm = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    token=token,
                    use_cache= True,
                )
        return llm
    
  
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        labels = kwargs["labels"]
    
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output