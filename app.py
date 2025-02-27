import streamlit as st
import torch

# Load vocabulary
vocab = torch.load("https://github.com/haidermb25/vocab.git/vocab.pth", weights_only=False)  # Make sure this file exists in the correct path

# Load Transformer model
model = torch.load("https://github.com/haidermb25/AI-Model-C-Code.git/transformer_seq2seq.pth", map_location=torch.device("cpu"))
model.eval()  # Set model to evaluation mode

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
