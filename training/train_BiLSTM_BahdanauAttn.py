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

# ==== Optimized Bahdanau Attention ====
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim * 2, hidden_dim)  # [h;e] -> hidden_dim
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(0).repeat(seq_len, 1, 1)  # [seq_len, batch_size, hidden_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [seq_len, batch_size, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [seq_len, batch_size]
        return F.softmax(attention, dim=0)  # [seq_len, batch_size]

# ==== BiLSTM Model with Bahdanau Attention ====
class CpGPredictor(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=64, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded = self.layernorm(embedded)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # Concatenate forward and backward hidden states
        attn_weights = self.attention(hidden, lstm_out.transpose(0, 1))
        attn_weights = attn_weights.transpose(0, 1).unsqueeze(1)
        context = torch.bmm(attn_weights, lstm_out)
        
        out = self.dropout(context.squeeze(1))
        prediction = self.classifier(out).squeeze()
        return context.squeeze(1), prediction, attn_weights.squeeze(1)

# ==== Training Function ====
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, weight_decay=1e-5, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            _, out, _ = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        preds, actuals = [], []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                _, out, _ = model(x, lengths)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds.extend(out.cpu().numpy())
                actuals.extend(y.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)

        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | R²: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, r2, mae, mse])

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./models/bilstm_bahdanau_checkpoint.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    torch.save(model.state_dict(), "./models/bilstm_bahdanau_final.pt")
    return train_losses, val_losses, best_val_loss

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

# ==== Global Data Preparation ====
data = create_dataset(5000, seq_len=200)
sequences, labels = zip(*data)
train_data, temp_data, train_labels, temp_labels = train_test_split(sequences, labels, test_size=0.3, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# ==== Main Workflow ====
def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./training", exist_ok=True)

    # Hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_trial.params
    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Best validation loss: {study.best_value}")

    # Final training with best parameters
    train_loader = DataLoader(DNASequencesDataset(train_data, train_labels), batch_size=best_params["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(DNASequencesDataset(val_data, val_labels), batch_size=best_params["batch_size"], collate_fn=collate_fn)
    test_loader = DataLoader(DNASequencesDataset(test_data, test_labels), batch_size=best_params["batch_size"], collate_fn=collate_fn)

    model = CpGPredictor(input_dim=5, embedding_dim=best_params["embedding_dim"],
                         hidden_dim=best_params["hidden_dim"], num_layers=best_params["num_layers"],
                         dropout=best_params["dropout"]).to(device)

    train_losses, val_losses, best_val_loss = train_model(model, train_loader, val_loader, lr=best_params["lr"],
                                                          weight_decay=best_params["weight_decay"])
    logging.info(f"Best validation loss achieved: {best_val_loss:.4f}")

    # Evaluate on test set
    criterion = nn.MSELoss()
    model.eval()
    test_loss, test_preds, test_targets = 0, [], []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            _, out, _ = model(x, lengths)
            loss = criterion(out, y)
            test_loss += loss.item()
            test_preds.extend(out.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
    test_loss /= len(test_loader)
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_mse = mean_squared_error(test_targets, test_preds)
    logging.info(f"Test Loss: {test_loss:.4f} | Test R²: {test_r2:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f}")

    # Example prediction and attention visualization
    test_seq = "ATGCGCGTANCGCCGNCCGGCGCGTANCTACGGCGCGTANCCGCGTANCGCCGNCCGGCGCGTANCTANCGCGGCGCGTAGCGTANCCGCGTANNCCGCGTANCAT"
    pred, attn = predict_cpg(model, test_seq, return_attention=True)
    actual = count_cpgs(test_seq)
    logging.info(f"Prediction for {test_seq}: {pred:.2f}, Actual: {actual}")
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(attn)), attn)
    plt.title(f"Attention Weights for {test_seq}")
    plt.xlabel("Position")
    plt.ylabel("Attention Weight")
    plt.savefig("./training/attention_weights.png")
    plt.close()

    # Plot training results
    plot_results(train_losses, val_losses)

if __name__ == "__main__":
    main()