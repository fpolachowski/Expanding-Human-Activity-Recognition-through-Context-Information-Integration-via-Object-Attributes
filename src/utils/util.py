import os
import torch.nn.functional as F


def safe_create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def n_params(model):
    """ Calculate total number of parameters in a model.
    Args:
        model: nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_cosine_eval_accuracy(x, y, labels):
    similarity = F.cosine_similarity(x, y, dim=-1)
    matching = similarity.argmax(dim=1) == labels
    accuracy = matching.sum() / len(similarity)
    return accuracy

def calculate_top_N_cosine_eval_accuracy(x, y, labels, N):
    similarity = F.cosine_similarity(x, y, dim=-1)
    _, indices = similarity.topk(N)
    labels = labels.unsqueeze(1)
    matching = indices == labels
    accuracy = matching.sum() / len(similarity)
    return accuracy
