import streamlit as st
import torch
import urllib.request
from transformer_model import TransformerSeq2Seq  # Import your model class

# URLs of the files
vocab_url = "https://raw.githubusercontent.com/haidermb25/vocab/main/vocab.pth"
model_url = "https://raw.githubusercontent.com/haidermb25/AI-Model-C-Code/main/transformer_seq2seq.pth"

# Local filenames
vocab_file = "vocab.pth"
model_file = "transformer_seq2seq.pth"

# Download the files
urllib.request.urlretrieve(vocab_url, vocab_file)
urllib.request.urlretrieve(model_url, model_file)

# Load vocabulary
vocab = torch.load(vocab_file, map_location=torch.device("cpu"))  

# Load model weights properly
state_dict = torch.load(model_file, map_location=torch.device("cpu"))  

# Check if state_dict is an OrderedDict (weights only)
if isinstance(state_dict, torch.nn.Module):
    model = state_dict  # If entire model is saved, use it directly
else:
    model = TransformerSeq2Seq()  # Initialize model
    model.load_state_dict(state_dict)  # Load weights into model

# Set model to evaluation mode
model.eval()

# Function to encode input sentence
def encode_input(sentence):
    tokens = ["<sos>"] + sentence.lower().split() + ["<eos>"]
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]  
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  

# Function to generate code
def predict(input_sentence, max_length=50):
    src = encode_input(input_sentence)
    tgt = torch.tensor([[vocab["<sos>"]]], dtype=torch.long)  

    for _ in range(max_length):
        output = model(src, tgt)
        next_token = output.argmax(dim=-1)[:, -1].item()  

        if next_token == vocab["<eos>"]:  
            break

        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    output_tokens = [list(vocab.keys())[idx] for idx in tgt.squeeze(0).tolist()]
    return " ".join(output_tokens[1:])  

# Streamlit UI
st.title("Pseudocode to Code Generator 🚀")
st.write("Enter your pseudocode below:")

# User input
input_text = st.text_area("Pseudocode:", "create string s")

# Generate button
if st.button("Generate Code"):
    output_text = predict(input_text)
    st.subheader("Generated Code:")
    st.code(output_text, language="python")
