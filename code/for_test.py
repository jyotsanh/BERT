import math
import numpy as np
from bpemb import BPEmb
import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask=None):
#below key_dim is the dimension for key which will be used for scaling the matmul of query and key.
  key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
  print(f"shape of key : {tf.shape(key)}")
  print(f"shape of query : {tf.shape(query)}")

  #multiplication between query and key which will be divided by square root of key_dim 
  scaled_scores = tf.matmul(query, key, transpose_b=True) / np.sqrt(key_dim)
  print(f"scaled  Attention scores [querty x key] : {scaled_scores}")
  if mask is not None:
    scaled_scores = tf.where(mask==0, -np.inf, scaled_scores)

  # softmax layer
  softmax = tf.keras.layers.Softmax()
  
  # scaled scores  goes through softmax layer..
  weights = softmax(scaled_scores) 
  
  print(f"Attention Weights : {weights}")
  
  print(f"shape of value : {tf.shape(value)}")
  print(f"shape of weights : {tf.shape(weights)}")
  # returns the multiplication of scaled scores and value 
  return tf.matmul(weights, value), weights



batch_size = 1  #includes the number of batches
seq_len = 3  # number of vectors aka sequence
embed_dim = 12 #dimension of seq vector aka embed dimension
num_heads = 3 #number of self-attention head
head_dim = embed_dim // num_heads  #dimension for query , key and value 

print(f"Dimension of each head: {head_dim}")

x = np.random.rand(batch_size, seq_len, embed_dim).round(1)
print("Input shape: ", x.shape, "\n")

print("Input:\n", x)

# The query weights for each head.
wq0 = np.random.rand(embed_dim, head_dim).round(1)
wq1 = np.random.rand(embed_dim, head_dim).round(1)
wq2 = np.random.rand(embed_dim, head_dim).round(1)

# The key weights for each head. 
wk0 = np.random.rand(embed_dim, head_dim).round(1)
wk1 = np.random.rand(embed_dim, head_dim).round(1)
wk2 = np.random.rand(embed_dim, head_dim).round(1)

# The value weights for each head.
wv0 = np.random.rand(embed_dim, head_dim).round(1)
wv1 = np.random.rand(embed_dim, head_dim).round(1)
wv2 = np.random.rand(embed_dim, head_dim).round(1)


print("The three sets of query weights (one for each head):")
print("wq0:\n", wq0)
print("wq1:\n", wq1)
print("wq2:\n", wq1)

# Geneated queries, keys, and values for the first head.
q0 = np.dot(x, wq0)
k0 = np.dot(x, wk0)
v0 = np.dot(x, wv0)

# Geneated queries, keys, and values for the second head.
q1 = np.dot(x, wq1)
k1 = np.dot(x, wk1)
v1 = np.dot(x, wv1)

# Geneated queries, keys, and values for the third head.
q2 = np.dot(x, wq2)
k2 = np.dot(x, wk2)
v2 = np.dot(x, wv2)


print(" ")
print("Q, K, and V for first head:\n")

print(f"q0 {q0.shape}:\n", q0, "\n")
print(f"k0 {k0.shape}:\n", k0, "\n")
print(f"v0 {v0.shape}:\n", v0)

out0, attn_weights0 = scaled_dot_product_attention(q0, k0, v0)
out1, attn_weights1 = scaled_dot_product_attention(q1, k1, v1)
out2, attn_weights2 = scaled_dot_product_attention(q2, k2, v2)


# --------------------------------------------------------------------
# print("Output from first attention head: ", out0, "\n")
# print("Attention weights from first head: ", attn_weights0)
# ---------------------------------------------------------------------

combined_out_a = np.concatenate((out0, out1, out2), axis=-1)
print(f"Combined output from all heads {combined_out_a.shape}:")
print(combined_out_a)

# The final step would be to run combined_out_a through a linear/dense layer 
# for further processing.




# Let's now get the same thing done using a single query weight matrix, single key weight matrix, and single value weight matrix.
# These were our separate per-head query weights:
wq = np.concatenate((wq0, wq1, wq2), axis=1)
wk = np.concatenate((wk0, wk1, wk2), axis=1)
wv = np.concatenate((wv0, wv1, wv2), axis=1)
print(f"Single query weight matrix {wq.shape}: \n", wq)

k_s = np.dot(x, wk)
v_s = np.dot(x, wv)
q_s = np.dot(x, wq)
print(f"Query vectors using a single weight matrix {q_s.shape}:\n", q_s)