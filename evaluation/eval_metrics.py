import numpy as np

def precision_at_k(predicted, relevant, k):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(relevant)) / k

def recall_at_k(predicted, relevant, k):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(relevant)) / len(relevant)

def ndcg_at_k(predicted, relevant, k):
    def dcg(scores):
        return np.sum((2 ** np.array(scores) - 1) / np.log2(np.arange(2, len(scores) + 2)))
    scores = [1 if p in relevant else 0 for p in predicted[:k]]
    ideal = sorted(scores, reverse=True)
    return dcg(scores) / dcg(ideal) if np.sum(ideal) else 0.0
