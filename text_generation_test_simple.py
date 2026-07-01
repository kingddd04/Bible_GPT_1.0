import torch
from gpt_tokenizer import GptTokenizer
from llm_gpt import GptModel, GPTConfig
from gpt_tokenizer import GptTokenizer   

def load_model(checkpoint_path="best_model.pt"):
    config = GPTConfig(vocab_size=50257,block_size=256,n_layer=4,n_head=4,_embd=256,dropout=0.1 )
    
    # Model Creation
    model = GptModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100):
    model.eval()

    # Encode prompt
    x = tokenizer.to_tensor(prompt).unsqueeze(0)  

    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = model(x, None)

        # Take last token's logits
        logits = logits[:, -1, :]

        # Softmax → probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        x = torch.cat([x, next_token], dim=1)

    # Decode final sequence
    return tokenizer.decode(x[0])


model = load_model("bible_gpt.pt")
tokenizer = GptTokenizer("gpt2")

prompt = "jesus is the"
output = generate(model, tokenizer, prompt, max_new_tokens=7)

print(output)
