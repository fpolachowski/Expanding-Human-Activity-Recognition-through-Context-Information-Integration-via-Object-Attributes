import os
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import utils.box_ops as box_ops


def safe_create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def n_params(model):
    """ Calculate total number of parameters in a model.
    Args:
        model: nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_dot_eval_accuracy(x, y, labels):
    similarity = torch.mm(x, y.transpose(1, 0))
    matching = similarity.argmax(dim=1) == labels
    accuracy = matching.sum() / len(similarity)
    return accuracy

def calculate_top_N_dot_eval_accuracy(x, y, labels, N):
    similarity = torch.mm(x, y.transpose(1, 0))
    _, indices = similarity.topk(N)
    labels = labels.unsqueeze(1)
    matching = indices == labels
    accuracy = matching.sum() / len(similarity)
    return accuracy
