import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from functools import partial
from typing import Sequence
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random
import numpy as np
import optuna
import os
import logging
import csv
from datetime import datetime

# ==== Mandatory Code (DO NOT CHANGE) ====
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        if dimer == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(5))}
int2dna = {i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# ==== Device Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Logging Setup ====
log_dir = "./logs/bilstm_bahdanau"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# ==== CSV Metrics File ====
metrics_csv = os.path.join(log_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_headers = ["Epoch", "Train_Loss", "Val_Loss", "Val_R2", "Val_MAE", "Val_MSE"]
with open(metrics_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# ==== Dataset Creation ====
def create_dataset(n=5000, seq_len=200):
    """
    Generate synthetic DNA sequences and their CG counts using mandatory functions.
    
    Args:
        n (int): Number of sequences to generate.
        seq_len (int): Length of sequences (default: 200).
    
    Returns:
        list: List of tuples (encoded_sequence, CG count).
    """
    data = []
    seq_generator = rand_sequence(n, seq_len)
    for int_seq in seq_generator:
        dna_seq = ''.join(intseq_to_dnaseq(int_seq))
        cg_count = count_cpgs(dna_seq)
        data.append((int_seq, cg_count))
    return data

# ==== PyTorch Dataset ====
class DNASequencesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return encoded, label

# ==== Collate Function for Padding ====
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_seqs, labels, lengths



# ==== Prediction Function ====
def predict_cpg(model, sequence, return_attention=False):
    """
    Predict CG count for a given DNA sequence and optionally return attention weights.
    
    Args:
        model (nn.Module): Trained CpGPredictor model.
        sequence (str or list): DNA sequence (str) or integer-encoded sequence (list).
        return_attention (bool): Whether to return attention weights.
    
    Returns:
        float or tuple: Predicted CG count, and optionally attention weights.
    """
    model.eval()
    if isinstance(sequence, str):
        int_seq = list(dnaseq_to_intseq(sequence.upper()))
    else:
        int_seq = sequence
    x = torch.tensor([int_seq], dtype=torch.long).to(device)
    lengths = torch.tensor([len(int_seq)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        _, pred, attn_weights = model(x, lengths)
    pred = pred.item()
    if return_attention:
        return pred, attn_weights.squeeze().cpu().numpy()
    return pred

# ==== Optuna Objective Function ====
def objective(trial):
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    train_loader = DataLoader(DNASequencesDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(DNASequencesDataset(val_data, val_labels), batch_size=batch_size, collate_fn=collate_fn)

    model = CpGPredictor(input_dim=5, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers, dropout=dropout)
    _, _, best_val_loss = train_model(model, train_loader, val_loader, lr=lr, weight_decay=weight_decay)
    return best_val_loss

# ==== Plotting Function ====
def plot_results(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./training/loss_curve.png')
    plt.close()
