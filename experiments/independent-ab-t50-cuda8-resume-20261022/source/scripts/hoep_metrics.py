import torch

def summarize_group_logits(logits, labels, old_count):
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("logits and labels have incompatible shapes")
    if not 0 < old_count < logits.shape[1]:
        raise ValueError("old_count must be within the class range")
    labels = labels.long()
    prediction = logits.argmax(dim=1)
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logit - competitors.max(dim=1).values
    result = {}
    for name, mask in (("old", labels < old_count), ("new", labels >= old_count)):
        if not mask.any():
            raise ValueError(f"empty {name} evaluation group")
        result[name] = {
            "count": int(mask.sum()),
            "top1": 100.0 * float((prediction[mask] == labels[mask]).float().mean()),
            "mean_true_margin": float(margins[mask].mean()),
            "predicted_new_rate": 100.0 * float(
                (prediction[mask] >= old_count).float().mean()
            ),
        }
    return result

def paired_prediction_changes(before, after, labels, old_count):
    if before.shape != after.shape or len(labels) != len(before):
        raise ValueError("paired logits and labels must have matching shapes")
    before_pred = before.argmax(dim=1)
    after_pred = after.argmax(dim=1)
    result = {}
    for name, mask in (("old", labels < old_count), ("new", labels >= old_count)):
        result[name] = {
            "count": int(mask.sum()),
            "prediction_changed": int(((before_pred != after_pred) & mask).sum()),
            "correct_to_wrong": int(
                ((before_pred == labels) & (after_pred != labels) & mask).sum()
            ),
            "wrong_to_correct": int(
                ((before_pred != labels) & (after_pred == labels) & mask).sum()
            ),
        }
    return result
