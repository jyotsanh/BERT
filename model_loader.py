from transformers import BertModel, BertTokenizer
import torch

# Load pretrained model/tokenizer
tokenizer2 = BertTokenizer.from_pretrained('bert-large-uncased')
model2 = BertModel.from_pretrained('./best_models/model_1000') 



tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')  # replace with the path to your saved model

# Words to compare
word1 = "HOUSE"
word2 = "TYRELL"

# Tokenize input
input_ids1 = tokenizer.encode(word1, add_special_tokens=True)
input_ids2 = tokenizer.encode(word2, add_special_tokens=True)

input_ids1_1= tokenizer2.encode(word1, add_special_tokens=True)
input_ids2_2 = tokenizer2.encode(word2, add_special_tokens=True)


# Convert input to PyTorch tensors
input_ids1 = torch.tensor(input_ids1).unsqueeze(0)
input_ids2 = torch.tensor(input_ids2).unsqueeze(0)

# Convert input to PyTorch tensors
input_ids1_1 = torch.tensor(input_ids1_1).unsqueeze(0)
input_ids2_2 = torch.tensor(input_ids2_2).unsqueeze(0)

# Get embeddings
outputs1 = model(input_ids1)
outputs2 = model(input_ids2)

# Get embeddings
outputs1_1 = model2(input_ids1_1)
outputs2_1 = model2(input_ids2_2)



# Get the embeddings for the words
word1_embedding = outputs1[0][0][1]
word2_embedding = outputs2[0][0][1]




# Calculate Euclidean distance
euclidean_distance = torch.dist(word1_embedding, word2_embedding)

print(f"The Euclidean distance between the embeddings for '{word1}' and '{word2}' is {euclidean_distance.item()}")



# Get the embeddings for the words
word1_embedding = outputs1_1[0][0][1]
word2_embedding = outputs2_1[0][0][1]




# Calculate Euclidean distance
euclidean_distance = torch.dist(word1_embedding, word2_embedding)

print(f"The Euclidean distance between the embeddings for '{word1}' and '{word2}' is {euclidean_distance.item()}")
