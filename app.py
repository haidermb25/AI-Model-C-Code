import torch
import torch.nn as nn
from torchtext.vocab import Vocab
import spacy
import streamlit as st
import requests
import os

# File URLs (Replace with your actual GitHub RAW URLs if files are small)
VOCAB_URL = "https://raw.githubusercontent.com/haidermb25/vocab/main/vocab.pth"
MODEL_URL= "https://raw.githubusercontent.com/haidermb25/AI-Model-C-Code/main/transformer_seq2seq.pth"

# File Paths
VOCAB_PATH = "vocab.pth"
MODEL_PATH = "transformer_seq2seq.pth"

# Function to download files
def download_file(url, file_path):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading {file_path}..."):
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

# Download files if not present
download_file(VOCAB_URL, VOCAB_PATH)
download_file(MODEL_URL, MODEL_PATH)

# Load tokenizer
spacy_en = spacy.load("en_core_web_sm")

# Load saved vocab
vocab = torch.load(VOCAB_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# Load model
model = TransformerSeq2Seq(len(vocab)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Tokenizer
def tokenize(sentence):
    return [token.text for token in spacy_en(str(sentence).lower())]

# Encode input
def encode_input(text):
    tokens = tokenize(text)
    encoded = torch.tensor([vocab["<sos>"]] + [vocab[token] for token in tokens] + [vocab["<eos>"]], dtype=torch.long)
    return encoded.unsqueeze(1).to(device)  # Add batch dimension

# Decode output
def decode_output(output_tokens):
    return " ".join([vocab.lookup_token(idx) for idx in output_tokens if idx not in [vocab["<sos>"], vocab["<eos>"], vocab["<pad>"]]])

# Generate output
def generate_output(input_text, max_length=50):
    src = encode_input(input_text)
    tgt = torch.tensor([[vocab["<sos>"]]], dtype=torch.long).to(device)

    for _ in range(max_length):
        output = model(src, tgt)
        next_token = output.argmax(dim=-1)[-1, 0].item()
        if next_token == vocab["<eos>"]:
            break
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=0)

    return decode_output(tgt.squeeze().tolist())

# Streamlit UI
st.title("Transformer Text Generator")
input_text = st.text_input("Enter a sentence:")
if st.button("Generate"):
    output_text = generate_output(input_text)
    st.write(f"**Output:** {output_text}")
