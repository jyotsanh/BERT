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


class MultiHeadSelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadSelfAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    self.d_head = self.d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(self.d_model)
    self.wk = tf.keras.layers.Dense(self.d_model)
    self.wv = tf.keras.layers.Dense(self.d_model)

    # Linear layer to generate the final output.
    self.dense = tf.keras.layers.Dense(self.d_model)
  
  def split_heads(self, x):
    batch_size = x.shape[0]

    split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
    return tf.transpose(split_inputs, perm=[0, 2, 1, 3])
  
  def merge_heads(self, x):
    batch_size = x.shape[0]

    merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
    return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))

  def call(self, q, k, v, mask):
    qs = self.wq(q)
    ks = self.wk(k)
    vs = self.wv(v)

    qs = self.split_heads(qs)
    ks = self.split_heads(ks)
    vs = self.split_heads(vs)

    output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
    output = self.merge_heads(output)

    return self.dense(output), attn_weights




batch_size = 1  #includes the number of batches
seq_len = 3  # number of vectors aka sequence
embed_dim = 12 #dimension of seq vector aka embed dimension
num_heads = 3 #number of self-attention head
head_dim = embed_dim // num_heads  #dimension for query , key and value 

x = np.random.rand(batch_size, seq_len, embed_dim).round(1)
print("Input shape: ", x.shape, "\n")

print("Input:\n", x)

mhsa = MultiHeadSelfAttention(12, 3)

output, attn_weights = mhsa(x, x, x, None)
print(f"MHSA output{output.shape}:")
print(output)