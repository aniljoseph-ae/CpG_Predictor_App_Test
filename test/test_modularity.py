import os
import sys
import logging
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add root directory to sys.path for relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import custom modules (update the paths based on your structure)
from model_architecture.bilstm_bahdanu_model import CpGPredictor
from preprocess.bilstm_bahdanu_preprocess import predict_cpg, count_cpgs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_sequences(
    sequences,
    model_path=r"./models/bilstm_bahdanau_final.pt",
    output_dir="./logs/testing/testing_image",
    log_dir="./logs/testing/testing_logs",
    save_attention=True,
    save_csv=True
):
    """
    Predicts CpG counts for a list of sequences using a pre-trained model and logs results.

    Args:
        sequences (list[str]): List of DNA sequences.
        model_path (str): Path to the saved model.
        output_dir (str): Directory to save attention plots.
        log_dir (str): Directory to save log file.
        save_attention (bool): Whether to save attention bar plots.
        save_csv (bool): Whether to save results in CSV format.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup timestamped logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_log_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load the model
    model = CpGPredictor(
        input_dim=5,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.33
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("✅ Model loaded successfully.")

    # For CSV logging
    results = []

    for idx, seq in enumerate(sequences, 1):
        pred, attn = predict_cpg(model, seq, return_attention=True)
        actual = count_cpgs(seq)
        error = abs(pred - actual)

        logging.info(f"\nSequence {idx}:")
        logging.info(f"📌 {seq}")
        logging.info(f"✅ Actual CpG: {actual}, 🧠 Predicted CpG: {pred:.2f}, ❗ Error: {error:.2f}")

        results.append({
            "Sequence_ID": f"Seq_{idx}",
            "Sequence": seq,
            "Actual_CpG": actual,
            "Predicted_CpG": round(pred, 2),
            "Error": round(error, 2)
        })

        if save_attention:
            plt.figure(figsize=(14, 4))
            plt.bar(range(len(attn)), attn)
            plt.title(f"Attention - Seq {idx}\nActual: {actual}, Predicted: {pred:.2f}")
            plt.xlabel("Nucleotide Position")
            plt.ylabel("Attention Weight")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"attention_seq_{idx}.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"🖼️ Attention plot saved to: {plot_path}")

    # Save results to CSV
    if save_csv:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f"prediction_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"📝 Predictions saved to: {csv_path}")

    print(f"✅ Testing complete. Logs: {log_file}, Results: {csv_path if save_csv else 'Not saved'}")
