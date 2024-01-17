from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import nltk
import random


nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)


def mask_random_word(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)

    if not words:
        return sentence, sentence  # Return the original sentence if it's empty

    # Choose a random word to mask
    word_to_mask = random.choice(words)

    # Replace the chosen word with [MASK]
    masked_sentence = sentence.replace(word_to_mask, tokenizer.mask_token, 1)

    return masked_sentence, sentence

def process_book(filename):
    model.train()  # Set the model to training mode
    with open(filename, 'r') as file:
        for line in file:
            # Tokenize the line into sentences
            sentences = nltk.sent_tokenize(line)

            for sentence in sentences:
                masked_sentence, original_sentence = mask_random_word(sentence)
                inputs = tokenizer(masked_sentence, return_tensors='pt', padding=True, truncation=True)
                labels = tokenizer(original_sentence, return_tensors='pt', padding=True, truncation=True)["input_ids"]

                # Ensure the input and label tensors have the same shape
                if inputs["input_ids"].shape[1] != labels.shape[1]:
                    continue

                # Clear the gradients from the previous step
                optimizer.zero_grad()

                # Forward pass
                outputs = model(**inputs, labels=labels)
                # -> As you can see we input the model with double '**' sign it means it unpackas the dict like below:
# -> we can also pass the input like this model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
                
                loss = outputs.loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                print(loss.item())

process_book('/content/A-Game-Of-Thrones-Book.txt')