# TITLE: Test / Evaluate Model

# IMPORTS
import csv
import torch
import argparse
from model import RoCNN
from data_director import DataDirector
from eval_metrics import precision, recall, f1_score

def test(model, data_director, device):
    """ Evaluates RoCNN on a test (or validation) split.

    Arguments:
    model {RoCNN} -- Trained model with sigmoid output.
    data_director {DataDirector} -- Provides dataloaders; expects "tst" key. 
                                    If absent, uses "val".
    device {str} -- 'cuda' or 'cpu'.
    """
    # Loss function.
    bce_loss = torch.nn.BCELoss(reduction="none")

    # TEST
    tst_loss = 0.0
    tst_tp = 0
    tst_tn = 0
    tst_fp = 0
    tst_fn = 0
    tst_n = 0

    model.eval() # Set to evaluation mode.
    with torch.no_grad(): # No gradient tracking.
        # For each batch ...
        for y, x in data_director.dataloader["val"]:
            # Prepare data.
            x = x.to(device, dtype=torch.float32) # [B, 6, 64, 64]
            y = y.to(device, dtype=torch.float32) # [B] or [B,1]
            if y.ndim == 1: y = y.unsqueeze(1) # make [B,1]

            # Get predictions.
            p = model(x) # [B, 1] sigmoid probability predictions.

            # Compute loss.
            loss = bce_loss(p, y)

            # Convert probs to 0/1 preds.
            p_bin = (p >= 0.5).float() 
            
            # Update running metrics.
            tst_loss += loss.sum().item()
            tst_n += x.size(0)
            tst_tp += ((p_bin == 1) & (y == 1)).sum().item()
            tst_tn += ((p_bin == 0) & (y == 0)).sum().item()
            tst_fp += ((p_bin == 1) & (y == 0)).sum().item()
            tst_fn += ((p_bin == 0) & (y == 1)).sum().item() 

    # Compute epoch metrics.
    tst_loss = tst_loss / tst_n
    tst_p = precision(tst_tp, tst_fp)
    tst_r = recall(tst_tp, tst_fn)
    tst_f1 = f1_score(tst_p, tst_r)
    tst_acc = (tst_tp + tst_tn) / tst_n

    # LOG TO CONSOLE
    print("\n=== TEST SUMMARY ===")
    print(f"bce loss: {tst_loss:.4f}")
    print(f"precision: {tst_p:.4f} | recall: {tst_r:.4f} | f1: {tst_f1:.4f}")
    print(f"accuracy: {tst_acc:.4f}")
    print(f"counts: TP={tst_tp} TN={tst_tn} FP={tst_fp} FN={tst_fn} N={tst_n}")

    return {
        "split": "test",
        "bce_loss": tst_loss,
        "precision": tst_p,
        "recall": tst_r,
        "f1": tst_f1,
        "accuracy": tst_acc,
        "tp": tst_tp, "tn": tst_tn, "fp": tst_fp, 
        "fn": tst_fn, "n": tst_n
    }

def main():
    """Main function to load a checkpoint and evaluate RoCNN."""
    parser = argparse.ArgumentParser(description="Test / Evaluate RoCNN model.")
    parser.add_argument("-dd", "--dir_data", type=str, required=True,
                        help="Path to folder with test data (or val if no tst split).")
    parser.add_argument("-tp", "--trn_pc", type=float, required=True,
                        help="Training set %.")
    parser.add_argument("-vp", "--val_pc", type=float, required=True,
                        help="Validation set %.")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                        help="Mini-batch size. (Default: 16)")
    parser.add_argument("-cp", "--path_checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt) to evaluate.")
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda' or 'cpu'). (Default: auto-detect)")
    args = parser.parse_args()

    # DEVICE SET UP
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # INITIALIZE MODEL
    print("Initializing model.")
    model = RoCNN()

    # LOAD CHECKPOINT IF GIVEN
    print(f"Loading checkpoint from '{args.path_checkpoint}'.")
    model.load_state_dict(torch.load(args.path_checkpoint,
                                     map_location=device))
    model.to(device) # Move model to device.

    # INITIALIZE DATA DIRECTOR
    print("Initializing data director.")
    data_director = DataDirector(dir_data=args.dir_data,
                                 split_pc_trn=args.trn_pc,
                                 split_pc_val=args.val_pc,
                                 batch_size=args.batch_size)

    # TRAIN MODEL
    print("Starting training...")
    model = test(model=model,
                 data_director=data_director,
                 device=device)

    print(f"Testing complete.")

if __name__ == "__main__":
    main()