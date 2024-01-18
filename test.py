import nltk
import random
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import re
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

optimizer = AdamW(model.parameters(), lr=1e-5)



def mask_random_word(sentence):
    # -> This function takes sentence and MASK a random words in the sentence and return two sentence 
    # -> with masking and without masking.
    
    
    # Tokenize the sentence into words for choosing which words to MASK.
    sentence = sentence.lower()

    # Remove special characters
    # -> We don't remove special character because it affects the bert contextualized vector embeddings. 
    # sentence = re.sub(r'\W', ' ', sentence)
    words = tokenizer.tokenize(sentence)
    
    if not words:
        return sentence, sentence  # Return the original sentence if it's empty

    # Choose a random word to mask
    word_to_mask = random.choice(words)

    # Replace the chosen word with [MASK]
    masked_sentence = sentence.replace(word_to_mask, tokenizer.mask_token, 1)

    return masked_sentence, sentence


def main():
    filename = "./content/A-Game-Of-Thrones-Book.txt"
    with open(filename, 'r', encoding='utf-8') as file:
            i=0
            for line in file:
                #first line of text file is read
                i+=1
                
                print(f"Line no : {i} finished")
                sentences = nltk.sent_tokenize(line)
                
                #sentence is seperated above
                if not sentences:
                    # This is for empty list which usually created during sent tokenizing.
                    continue
                else:
                    
                    for sentence in sentences:
                        # -> During sentence tokeinzing . it comes in list of sentences so we iterate the list.
                        masked_sentence, original_sentence = mask_random_word(sentence)
                        # -> mask_random_word is func check the function.
                        
                        words = tokenizer.tokenize(sentence)
                        
                        inputs = tokenizer(masked_sentence, return_tensors='pt', padding=True, truncation=True)
                        labels = tokenizer(original_sentence, return_tensors='pt', padding=True, truncation=True)["input_ids"]
                    
                        # Ensure the input and label tensors have the same shape
                        if inputs["input_ids"].shape[1] != labels.shape[1]:
                            continue
                        
                        
                        
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(**inputs, labels=labels)
                        # -> As you can see we input the model with double '**' sign it means it unpackas the dict like below:
# -> we can also pass the input like this model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
                
                        loss = outputs.loss

                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()

                        print(f"Batch: {i}, Loss: {loss.item()}")

                if i % 1000 == 0 or i == 27499:
                    model.save_pretrained(f"./models/model_{i}")
                    print(f"Model saved at ./models/model_{i}")
                
                
# sentence = "Hello !, Brother's what's up ? what are you doing ?"

# print(sentence)
if __name__== "__main__":
    main()