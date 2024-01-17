import nltk
import random

filename = "./content/A-Game-Of-Thrones-Book.txt"
with open(filename, 'r', encoding='utf-8') as file:
        i=0
        for line in file:
            i+=1
            
            sentences = nltk.sent_tokenize(line)
            if not sentences:
                continue
            else:
                
                for sentence in sentences:
                    print(sentence)

            
            if i==6:
                break