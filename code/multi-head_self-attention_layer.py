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





seq_len = 3
embed_dim = 4

queries = np.random.rand(seq_len, embed_dim)
keys = np.random.rand(seq_len, embed_dim)
values = np.random.rand(seq_len, embed_dim)

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