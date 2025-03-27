
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
    '''
    Best_hyperparameters: {
    'embedding_dim': 64, 
    'hidden_dim': 256, 
    'num_layers': 2, 
    'dropout': 0.33262129231366233, 
    'lr': 0.0008572034020671933, 
    'weight_decay': 3.4321573869941195e-06, 
    'batch_size': 32}
    
    '''