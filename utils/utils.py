import numpy as np
import os
import random
import torch

def seed_everything(seed=42):
    random.seed(seed)             # Python的内置随机库
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)          # Numpy库
    if torch is not None:         # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)          # 如果使用GPU
        torch.cuda.manual_seed_all(seed)      # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clip_path_map(path):
    if path=="pathclip-base":
        return "load_weights/pathclip/pathclip-base.pt"
    
    if path=="conch":
        return "load_weights/conch/pytorch_model.bin"


def my_compute_metrics(p):
    predictions, labels = p
    # predictions = np.argmax(predictions, axis=-1)
    
    mask = labels != -100

    correct = (labels == predictions) & mask
    accuracy =  round(np.mean(correct) *100, 2)

    return {
        'accuracy': accuracy,
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """

    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids, labels