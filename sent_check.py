from transformers import BertModel, BertTokenizer
import torch

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('./models/model_27000') 

# replace with the path to your saved model

# List of words from the Game of Thrones universe
words = ["Jon", "Snow", "Daenerys", "Targaryen", "Arya", "Stark", "Jaime", "Lannister", "Tyrion", "Cersei", "Sansa", "Bran", "Drogo", "Khal", "Brienne", "Tarth"]

# Calculate embeddings for all words
embeddings = {}

for word in words:
    input_ids = tokenizer.encode(word, return_tensors='pt')
    outputs = model(input_ids)
    embeddings[word] = outputs[0][0][1]

# Calculate Euclidean distances between all pairs of words
for i in range(len(words)):
    for j in range(i+1, len(words)):
        distance = torch.dist(embeddings[words[i]], embeddings[words[j]]).item()
        if distance < 5:
            print(f"The Euclidean distance between '{words[i]}' and '{words[j]}' is {distance}")
