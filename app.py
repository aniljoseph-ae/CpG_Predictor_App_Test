import streamlit as st
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ==== Logging ====
logging.basicConfig(level=logging.INFO)

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Bahdanau Attention ====
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=0)

# ==== BiLSTM + Attention Model ====
class CpGPredictor(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=64, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.layernorm(self.embedding(x))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output_transposed = output.transpose(0, 1)
        attn_weights = self.attention(hidden_cat, output_transposed)
        attn_weights = attn_weights.permute(1, 0).unsqueeze(1)
        attn_applied = torch.bmm(attn_weights, output)

        out = self.dropout(attn_applied.squeeze(1))
        prediction = self.classifier(out).squeeze()
        return attn_applied.squeeze(1), prediction, attn_weights.squeeze(1)

# ==== Simple DNA to int conversion ====
def dnaseq_to_intseq(seq):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [mapping.get(nuc, 0) for nuc in seq]

# ==== Highlight CG Pairs ====
def highlight_cg(seq):
    highlighted = ""
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i:i+2].upper() == "CG":
            highlighted += f"<span style='color:red; font-weight:bold;'>CG</span>"
            i += 2
        else:
            highlighted += seq[i]
            i += 1
    return highlighted

# ==== Preprocess Sequence ====
def preprocess_sequence(dna_sequence):
    int_seq = list(dnaseq_to_intseq(dna_sequence.upper()))
    tensor = torch.tensor(int_seq, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(int_seq)]).to(device)
    return tensor, lengths

# ==== Load Model ====

model = CpGPredictor(
    input_dim=5, 
    embedding_dim=64, 
    hidden_dim=256, 
    num_layers=2, 
    dropout=0.33262129231366233).to(device)

model_path = "./models/bilstm_bahdanau_final.pt"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"{model_path} not found")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
logging.info("✅ Model loaded successfully.")

# ==== Streamlit App Layout ====
st.set_page_config(page_title="CpG Site Predictor", layout="wide")
st.title("🔬 CpG Site Predictor using BiLSTM + Bahdanau Attention")

user_input = st.text_area("Enter a DNA sequence (A, C, G, T):", "", height=150)

if st.button("Predict"):
    try:
        if not user_input:
            st.warning("Please enter a DNA sequence.")
        else:
            # Display Highlighted Input
            highlighted_seq = highlight_cg(user_input)
            st.markdown(f"### Sequence with highlighted CG pairs:")
            st.markdown(f"<p style='font-family:monospace; font-size:16px;'>{highlighted_seq}</p>", unsafe_allow_html=True)

            # Count CG Pairs
            cg_count = sum(1 for i in range(len(user_input)-1) if user_input[i:i+2].upper() == "CG")
            st.info(f"📌 Actual CG pair count: **{cg_count}**")

            # Preprocess & Predict
            seq_tensor, lengths = preprocess_sequence(user_input)
            with torch.no_grad():
                _, pred, attn = model(seq_tensor, lengths)

            st.success(f"🧬 **Predicted CpG count:** {pred.item():.2f}")

            if st.checkbox("🔍 Show Attention Weights"):
                attn_weights = attn.squeeze().cpu().numpy()
                fig, ax = plt.subplots(figsize=(12, 2))
                ax.bar(np.arange(len(attn_weights)), attn_weights, color='lightcoral')
                ax.set_title("Attention Weights over Sequence")
                ax.set_xlabel("Nucleotide Position")
                ax.set_ylabel("Attention")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
