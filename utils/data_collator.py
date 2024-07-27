
from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch, _numpy_collate_batch
import numpy as np

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    image_processor = None
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def find_sequential_positions(self, large_list, elements_to_find):
        num_elements = len(elements_to_find)
        current_element_index = 0
        
        for i, value in enumerate(large_list):
            if value == elements_to_find[current_element_index]:
                current_element_index += 1
                if current_element_index == num_elements:
                    return i - num_elements + len(elements_to_find)
        return len(large_list)

    def __post_init__(self): 
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        labels = batch["input_ids"].clone()

        if self.tokenizer.bos_token_id is not None and self.tokenizer.eos_token_id is not None:
            # symbol =  [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('Ġ'), self.tokenizer.bos_token_id, self.tokenizer.convert_tokens_to_ids('ĠAnswer')]
            symbol =  [128009, 220, 128000, 22559]
            new_labels = []
            for label in labels:
                label = label.tolist()
                index = self.find_sequential_positions(label, symbol) + 1
                label[:index] = [-100] * index
                new_labels.append(torch.LongTensor(label))
        
        labels = torch.stack(new_labels)
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

