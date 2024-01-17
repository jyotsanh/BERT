from transformers import BertTokenizer, BertForMaskedLM, AdamW

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize input
input_ids = tokenizer.encode("Winter Stark", add_special_tokens=True)

# Convert input to PyTorch tensors
input_ids = torch.tensor(input_ids).unsqueeze(0)

# Get embeddings
outputs = model(input_ids)

# Get the embeddings for "Jon" and "Aegon"
jon_embedding = outputs[0][0][1]
aegon_embedding = outputs[0][0][2]

# Calculate Euclidean distance
euclidean_distance = torch.dist(jon_embedding, aegon_embedding)

print(f"The Euclidean distance between the embeddings for 'Jon' and 'Aegon' is {euclidean_distance.item()}")
