import streamlit as st
import torch
import urllib.request
from transformer_model import TransformerSeq2Seq  # Import your Transformer model class

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
vocab = torch.load(vocab_file, weights_only=False)  # Make sure this file exists in the correct path

# Initialize Transformer model
model = TransformerSeq2Seq()  # Replace with your actual model class

# Load weights into the model
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

# Set model to evaluation mode
model.eval()

# Function to encode input sentence
def encode_input(sentence):
    tokens = ["<sos>"] + sentence.lower().split() + ["<eos>"]
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]  # Handle unknown words
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Function to generate code
def predict(input_sentence, max_length=50):
    src = encode_input(input_sentence)
    tgt = torch.tensor([[vocab["<sos>"]]], dtype=torch.long)  # Start sequence with <sos>

    for _ in range(max_length):
        output = model(src, tgt)
        next_token = output.argmax(dim=-1)[:, -1].item()  # Get the most probable token

        if next_token == vocab["<eos>"]:  # Stop if <eos> token is generated
            break

        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    output_tokens = [list(vocab.keys())[idx] for idx in tgt.squeeze(0).tolist()]
    return " ".join(output_tokens[1:])  # Remove <sos> token

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
