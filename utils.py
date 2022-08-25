import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm


def to_var(x, requires_grad=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            label = data['labels']
            # print(label)
            scores = model(x_var).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))
    
    return acc


### Check model accuracy on model based on triggered dataset
def test_trigger(model, loader, target, batch):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    label = torch.zeros(batch)
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            label[:] = target   # setting all the target to target class
            scores = model(x_var).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()


    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the triggered data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


