import math
import numpy as np
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    key_dim = key.size(-1)
    print(f"shape of key : {key.size()}")
    print(f"shape of query : {query.size()}")

    scaled_scores = torch.matmul(query, key.T) / math.sqrt(key_dim)
    print(f"Scaled Attention scores [query x key] : {scaled_scores}")

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scaled_scores, dim=-1)
    print(f"Attention Weights : {weights}")

    print(f"shape of value : {value.size()}")
    print(f"shape of weights : {weights.size()}")

    output = torch.matmul(weights, value)
    return output, weights

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 3
embed_dim = 4

queries = torch.rand(seq_len, embed_dim, device=device)
keys = torch.rand(seq_len, embed_dim, device=device)
values = torch.rand(seq_len, embed_dim, device=device)

print(f"Queries : {queries}")
print("-----------------------------")
print(f"Keys : {keys}")
print("-----------------------------")
print(f"Values : {values}")

output, attn_weights = scaled_dot_product_attention(queries, keys, values)
print("-----------------------------")
print("Output\n", output, "\n")
print("-----------------------------")
print("Weights\n", attn_weights)
