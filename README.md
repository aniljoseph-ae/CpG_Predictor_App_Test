# CpG Site Predictor: BiLSTM with Bahdanau Attention

## Project Overview

The **CpG Site Predictor** is a deep learning-based tool designed to count the number of CpG sites (consecutive CG dinucleotides) in DNA sequences of varying lengths. The model leverages a Bidirectional Long Short-Term Memory (BiLSTM) network augmented with Bahdanau Attention to capture long-term dependencies and contextual information within DNA sequences. The project includes a training pipeline implemented in PyTorch and a user-friendly Streamlit web application for inference.

### Problem Statement
Given a DNA sequence composed of nucleotides (N, A, C, G, T), the task is to accurately count the number of CpG sites (consecutive CG pairs). The solution must handle variable-length sequences and provide high-confidence predictions.

### Key Features
- **Model Architecture**: BiLSTM with Bahdanau Attention for sequence modeling and context-aware prediction.
- **Training**: Optimized using Optuna for hyperparameter tuning, achieving a Test R² of 0.9977.
- **Deployment**: Streamlit app for interactive DNA sequence input and visualization of predictions and attention weights.
- **Scalability**: Handles variable-length sequences using PyTorch's `pack_padded_sequence` and `pad_packed_sequence`.

### Repository and Deployment
- **GitHub Repository**: [https://github.com/aniljoseph-ae/CpG_Predictor_App_Test/](https://github.com/aniljoseph-ae/CpG_Predictor_App_Test/)
- **Streamlit App**: [https://aniljoseph-ae-cpg-predictor-app-test-app-boble6.streamlit.app/](https://aniljoseph-ae-cpg-predictor-app-test-app-boble6.streamlit.app/)

---

## Model Architecture

The core of the CpG Site Predictor is the `CpGPredictor` class, a PyTorch `nn.Module` that integrates embedding, BiLSTM, attention, and classification layers.

### Components

1. **Embedding Layer**
   - **Input**: Integer-encoded DNA sequence (0: N, 1: A, 2: C, 3: G, 4: T).
   - **Parameters**: `input_dim=5`, `embedding_dim=64` (configurable via Optuna).
   - **Functionality**: Maps each nucleotide to a dense vector representation.
   - **Padding**: Uses `padding_idx=0` to handle variable-length sequences.

2. **Layer Normalization**
   - **Parameters**: Normalizes embeddings across the `embedding_dim` dimension.
   - **Purpose**: Stabilizes training by reducing internal covariate shift.

3. **Bidirectional LSTM (BiLSTM)**
   - **Parameters**: 
     - `hidden_dim=256` (configurable).
     - `num_layers=2` (configurable).
     - `dropout=0.3` (configurable, applied if `num_layers > 1`).
     - `bidirectional=True`.
   - **Input**: Normalized embeddings.
   - **Output**: Sequence of hidden states capturing forward and backward dependencies.
   - **Sequence Handling**: Uses `pack_padded_sequence` and `pad_packed_sequence` to efficiently process variable-length sequences.

4. **Bahdanau Attention**
   - **Class**: `Attention(nn.Module)`.
   - **Parameters**: 
     - Input to `attn`: Concatenation of hidden states (`hidden_dim * 4`) mapped to `hidden_dim`.
     - Output: Attention weights via `softmax`.
   - **Functionality**: Computes context-aware weights over the BiLSTM outputs, focusing on relevant positions for CpG prediction.
   - **Implementation**: 
     - Energy computation: `torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))`.
     - Attention scores: `F.softmax(self.v(energy), dim=0)`.

5. **Classifier**
   - **Parameters**: Linear layer mapping `hidden_dim * 2` (bidirectional hidden states) to a scalar output.
   - **Dropout**: Applied with `dropout=0.3` (configurable) to prevent overfitting.
   - **Output**: Predicted number of CpG sites (continuous value).

### Forward Pass
- **Input**: `(x, lengths)` where `x` is a tensor of shape `(batch_size, max_seq_len)` and `lengths` is a tensor of sequence lengths.
- **Steps**:
  1. Embed `x` and apply layer normalization.
  2. Pack embeddings and pass through BiLSTM.
  3. Unpack BiLSTM outputs and concatenate the final forward and backward hidden states.
  4. Apply Bahdanau Attention to weigh BiLSTM outputs.
  5. Compute context vector via batch matrix multiplication.
  6. Pass through dropout and classifier to obtain the prediction.
- **Output**: `(context, prediction, attention_weights)`.

---

## Training Pipeline

### Data Generation
- **Function**: `create_dataset(n=5000, seq_len=200)`.
- **Process**: Generates synthetic DNA sequences using `rand_sequence` and computes CpG counts with `count_cpgs`.
- **Output**: List of `(int_seq, cg_count)` pairs.

### Dataset and DataLoader
- **Class**: `DNASequencesDataset(Dataset)`.
- **Collate Function**: `collate_fn` pads sequences and returns `(padded_seqs, labels, lengths)`.
- **Splitting**: 70% train, 15% validation, 15% test (via `train_test_split`).

### Hyperparameter Optimization
- **Tool**: Optuna.
- **Search Space**:
  - `embedding_dim`: [32, 64, 128].
  - `hidden_dim`: [128, 256].
  - `num_layers`: [1, 2, 3].
  - `dropout`: [0.2, 0.5].
  - `lr`: [1e-4, 1e-3] (log scale).
  - `weight_decay`: [1e-6, 1e-4] (log scale).
  - `batch_size`: [32, 64].
- **Objective**: Minimize validation MSE loss.
- **Trials**: 20.

### Training Loop
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam with learning rate and weight decay from Optuna.
- **Scheduler**: `ReduceLROnPlateau` (patience=2, factor=0.5).
- **Early Stopping**: Patience of 5 epochs.
- **Gradient Clipping**: `max_norm=1.0`.
- **Metrics**: Train loss, validation loss, R², MAE, MSE logged per epoch.
  ![image](https://github.com/user-attachments/assets/3c0981cd-ef42-470c-9951-0048815e51ca)


### Performance
- **Test Metrics**:
  - Test Loss: 0.0165
  - Test R²: 0.9977
  - Test MAE: 0.0955
  - Test MSE: 0.0166
- **Interpretation**: The model achieves near-perfect prediction accuracy (R² ≈ 1) with minimal error, indicating robust generalization.

---

## Streamlit Application

### Functionality
- **Input**: User-provided DNA sequence via text area.
- **Output**:
  - Highlighted sequence with CG pairs in red.
  - Actual CpG count.
  - Predicted CpG count.
  - Optional attention weights visualization (bar plot).
  ![image](https://github.com/user-attachments/assets/f6ae31a1-b8df-4067-9e88-9107cce38245)

### Implementation
- **Preprocessing**: `dnaseq_to_intseq` converts DNA to integers; `preprocess_sequence` creates tensors.
- **Model Loading**: Loads pre-trained weights from `./models/bilstm_bahdanau_final.pt`.
- **Prediction**: Uses `model.eval()` with `torch.no_grad()` for inference.
- **Visualization**: Matplotlib for attention weights plot.

### Technical Details
- **Device**: Automatically selects CUDA or CPU.
- **Error Handling**: Checks for model file existence and valid user input.

---

## Evaluation and Results

### Model Strengths
- **High Accuracy**: R² of 0.9977 demonstrates excellent predictive power.
- **Variable-Length Handling**: Effective use of padding and packing ensures robustness.
- **Attention Mechanism**: Provides interpretability by highlighting influential sequence positions.

### Limitations
- **Synthetic Data**: Trained on randomly generated sequences, which may not fully represent biological variability.
- **Continuous Output**: Predicts a float value, requiring rounding for exact counts in some applications.

### Future Improvements
- **Real Data**: Incorporate biological DNA sequences (e.g., from genomic databases).
- **Classification Head**: Modify output to predict integer counts directly.
- **Multi-Task Learning**: Extend to detect other dinucleotide patterns.

---

## Conclusion

The CpG Site Predictor is a sophisticated application of BiLSTM and Bahdanau Attention, achieving exceptional performance on the task of counting CpG sites in DNA sequences. Its deployment via Streamlit makes it accessible to non-technical users, while the underlying PyTorch implementation and Optuna optimization ensure a robust and scalable solution. This project serves as a strong foundation for further bioinformatics applications.

--- 
