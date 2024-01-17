from transformers import BertModel, BertTokenizer
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('./best_models/model_1000')  # replace with the path to your saved model
bert_model = BertModel.from_pretrained('bert-large-uncased')
# Tokenize input
word1=input("Enter word : ")
word2 = input("Enter another : ")

input_ids = tokenizer.encode(f"{word1} {word2}", add_special_tokens=True)

# Convert input to PyTorch tensors
input_ids = torch.tensor(input_ids).unsqueeze(0)

# Get embeddings
outputs = model(input_ids)



# Get the embeddings for "Jon" and "Aegon"
jon_embedding = outputs[0][0][1]
aegon_embedding = outputs[0][0][2]

# Calculate Euclidean distance

euclidean_distance = torch.dist(jon_embedding, aegon_embedding)

print(f"The Euclidean distance between the embeddings for '{word1}' and '{word2}' is {euclidean_distance.item()}")


outputs = bert_model(input_ids)
# Get the embeddings for "Jon" and "Aegon"
jon_embedding = outputs[0][0][1]
aegon_embedding = outputs[0][0][2]

# Calculate Euclidean distance

euclidean_distance = torch.dist(jon_embedding, aegon_embedding)
print(f"The Euclidean distance between the embeddings for '{word1}' and '{word2}' for not trained model is {euclidean_distance.item()}")