# TITLE: Train & Validate Model

# IMPORTS
import time
import torch
import argparse
from model import RoCNN
from data_director import DataDirector
from eval_metrics import precision, recall, f1_score

def train_validate(model, data_director, epochs, lr,
                   save_path, log_path, device=None):
    """ Trains RoCNN for a binary task.
    Arguments:
    model {RoCNN} -- Model to train.
    data_director {DataDirector} -- Data Director with dataloaders.
    epochs {int} -- Number of training epochs.
    lr {float} -- Learning rate.
    save_path {str} -- Path to save best model weights.
    log_path {str} -- Path to save training validation log.
    device {str} -- Device to use (e.g., 'cuda' or 'cpu'). This is
                    automatically detected if None. (Default: None)
    """
    # Check if GPU is available. Move model to gpu if possible.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # RoCNN ends with Sigmoid(), so use BCELoss which 
    # expects probs in [0, 1].
    bce_loss = torch.nn.BCELoss(reduction="none")

    # Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep track of best validation loss.
    best_val_loss = float("inf")

    # Open logs.
    f_log = open(log_path, "w")
    f_log.write("epoch,trn_loss,val_loss,trn_f1,val_f1,trn_acc,val_acc\n")

    try:
        # For each epoch ...
        for epoch in range(1, epochs + 1):
            # TRAIN
            model.train() # Set to training mode.
            
            trn_loss = 0.0
            trn_tp = 0
            trn_tn = 0
            trn_fp = 0
            trn_fn = 0
            trn_n = 0
            
            # For each batch ...
            for y, x in data_director.dataloader["trn"]:
                # Prepare data.
                x = x.to(device, dtype=torch.float32) # [B, 6, 64, 64]
                y = y.to(device, dtype=torch.float32) # [B] or [B,1]
                if y.ndim == 1: y = y.unsqueeze(1) # make [B,1]

                # Zero (previous) gradients.
                optimizer.zero_grad()
                
                # Get predictions.
                p = model(x) # [B, 1] sigmoid probability predictions.

                # Compute loss.
                loss = bce_loss(p, y)

                # Back-propagate.
                loss.mean().backward() 

                # Update weights.
                optimizer.step() 

                # Convert probs to 0/1 preds.
                p_bin = (p >= 0.5).float() 
                
                # Update running metrics.
                trn_loss += loss.sum().item()
                trn_n += x.size(0)
                trn_tp += ((p_bin == 1) & (y == 1)).sum().item()
                trn_tn += ((p_bin == 0) & (y == 0)).sum().item()
                trn_fp += ((p_bin == 1) & (y == 0)).sum().item()
                trn_fn += ((p_bin == 0) & (y == 1)).sum().item() 

            # Compute epoch metrics.
            trn_loss = trn_loss / trn_n
            trn_p = precision(trn_tp, trn_fp)
            trn_r = recall(trn_tp, trn_fn)
            trn_f1 = f1_score(trn_p, trn_r)
            trn_acc = (trn_tp + trn_tn) / trn_n

            # VALIDATE
            val_loss = 0.0
            val_tp = 0
            val_tn = 0
            val_fp = 0
            val_fn = 0
            val_n = 0

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
                    val_loss += loss.sum().item()
                    val_n += x.size(0)
                    val_tp += ((p_bin == 1) & (y == 1)).sum().item()
                    val_tn += ((p_bin == 0) & (y == 0)).sum().item()
                    val_fp += ((p_bin == 1) & (y == 0)).sum().item()
                    val_fn += ((p_bin == 0) & (y == 1)).sum().item() 

            # Compute epoch metrics.
            val_loss = val_loss / val_n
            val_p = precision(val_tp, val_fp)
            val_r = recall(val_tp, val_fn)
            val_f1 = f1_score(val_p, val_r)
            val_acc = (val_tp + val_tn) / val_n

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            # LOG
            print(f"\nEpoch {epoch:03d}\n\t",
                f"trn bce loss: {trn_loss:.4f} |",
                f"val bce loss: {val_loss:.4f}\n\t",
                f"trn f1 score: {trn_f1:.4f} |",
                f"val f1 score: {val_f1:.4f}\n\t",
                f"trn acc score: {trn_acc:.4f} |",
                f"val acc score: {val_acc:.4f}")
            
            f_log.write(f"{epoch},{trn_loss:.4f},{val_loss:.4f},"
                    f"{trn_f1:.4f},{val_f1:.4f},"
                    f"{trn_acc:.4f},{val_acc:.4f}\n")
    except Exception as e:
        raise Exception(f"Exception: {e}.")
    finally:
        f_log.close()

    # If we tracked a best model, load it back
    if best_val_loss < float("inf"):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
    
def main():
    """Main function to train and validate the RoCNN model."""

    # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Train and validate RoCNN model.")
    parser.add_argument("-dd", "--dir_data", type=str, required=True,
                        help="Path to folder with data.")
    parser.add_argument("-tp", "--trn_pc", type=float, required=True,
                        help="Training set %.")
    parser.add_argument("-vp", "--val_pc", type=float, required=True,
                        help="Validation set %.")
    parser.add_argument("-e", "--num_epochs", type=int, required=True,
                        help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, required=True,
                        help="Learning rate for optimizer.")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                        help="Mini-batch size. (Default: 16)")
    parser.add_argument("-cp", "--path_checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training from. (Default: None)")
    parser.add_argument("-sp", "--save_path", type=str, default="roc_best.pt",
                        help="Path to save best model weights. (Default: roc_best.pt)")
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda' or 'cpu'). (Default: auto-detect)")
    parser.add_argument("-lp", "--log_path", type=str, default="log.csv",
                        help="Path to save training validation log. (Default: log.csv)")
    args = parser.parse_args()

    # DEVICE SET UP
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # INITIALIZE MODEL
    print("Initializing model.")
    model = RoCNN()

    # LOAD CHECKPOINT IF GIVEN
    if args.path_checkpoint is not None:
        print(f"Loading checkpoint from '{args.path_checkpoint}'.")
        model.load_state_dict(torch.load(args.path_checkpoint, 
                                         map_location=device))

    # INITIALIZE DATA DIRECTOR
    print("Initializing data director.")
    data_director = DataDirector(dir_data=args.dir_data,
                                 split_pc_trn=args.trn_pc,
                                 split_pc_val=args.val_pc,
                                 batch_size=args.batch_size)

    # TRAIN MODEL
    print("Starting training...")
    model = train_validate(model=model,
                           data_director=data_director,
                           epochs=args.num_epochs,
                           lr=args.learning_rate,
                           device=device,
                           save_path=args.save_path,
                           log_path=args.log_path)

    print(f"Training complete. Best model saved to '{args.save_path}'.")

if __name__ == "__main__":
    main()