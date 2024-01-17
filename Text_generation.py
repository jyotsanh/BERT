from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pretrained model and tokenizer
model = BertForMaskedLM.from_pretrained('./models/model_1000')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Starting prompt
prompt = "Jon Snow said, "

# Convert prompt to tokens
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
# Generate text
for _ in range(100):  # adjust the range for the length of text you want to generate
    # Predict the next token
    with torch.no_grad():
        output = model(input_ids)

    # Apply temperature to the output logits
    logits = output.logits[0, -1, :] 

    # Sample from the logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_token_id = torch.multinomial(probs, 1).item()

    # Append the top token ID to the input
    input_ids = torch.cat([input_ids, torch.tensor([[top_token_id]], device=device)], dim=1)# Move generated text back to CPU for decoding
generated_text = tokenizer.decode(input_ids[0].cpu())

print(generated_text)
