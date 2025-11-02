
# Simple greedy decode and CER computation for demo purposes.
import torch
import numpy as np
from utils.text_utils import TextMapper
def greedy_decode(logits, mapper=None):
    # logits: T x B x C (raw, not softmaxed)
    if isinstance(logits, torch.Tensor):
        probs = torch.argmax(logits, dim=2)  # T x B
        probs = probs.cpu().numpy().T  # B x T
    else:
        probs = np.argmax(logits, axis=2).T
    mapper = mapper or TextMapper()
    decoded = []
    for seq in probs:
        decoded.append(mapper.decode(seq.tolist()))
    return decoded

def compute_cer(pred, target):
    # simple CER (Levenshtein) - small demo implementation
    import numpy as np
    a = pred
    b = target
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1):
        dp[i,0] = i
    for j in range(len(b)+1):
        dp[0,j] = j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[len(a), len(b)] / max(1, len(b))
