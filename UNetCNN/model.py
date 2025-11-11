# IMPORTS
import os
import torch
import numpy as np
from torch.optim import Adam
from metrics import iou_score
import matplotlib.pyplot as plt

class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetSmall(torch.nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool = torch.nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        self.up3 = torch.nn.ConvTranspose2d(base_ch*8, base_ch*4, 
                                            kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = torch.nn.ConvTranspose2d(base_ch*4, base_ch*2, 
                                            kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = torch.nn.ConvTranspose2d(base_ch*2, base_ch, 
                                            kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        self.head = torch.nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.head(d1)
        return out  # logits
    
def train_model(model, train_loader, val_loader, device, epochs=10, 
                lr=1e-3, save_dir="checkpoints"):
    opt = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    os.makedirs(save_dir, exist_ok=True)

    best_val_iou = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        iou_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                iou_acc += iou_score(logits.detach().cpu(), 
                                     yb.detach().cpu()) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_iou = iou_acc / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f}",
              f"val_loss: {val_loss:.4f} val_iou: {val_iou:.4f}")

        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 
                       os.path.join(save_dir, "best_model.pth"))
    print("Training finished. Best val IoU:", best_val_iou)

def save_sample_predictions(model, dataset, device, 
                            out_dir="pred_samples", n_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i in range(n_samples):
            xb, yb = dataset[i]
            xb_t = xb.unsqueeze(0).to(device)
            logits = model(xb_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (probs > 0.5).astype(np.uint8)

            input_mask = xb.squeeze().numpy().astype(np.uint8)
            true_mask = yb.squeeze().numpy().astype(np.uint8)

            # Create a row of 3 images
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # adjust figsize as needed

            titles = [
                "Input timestep t",
                "True timestep (t + 1)",
                "Predicted timestep (t + 2)"
            ]
            images = [input_mask, true_mask, pred_bin]

            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img, cmap='gray')
                ax.set_title(title, fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{i:02d}.png"), 
                        bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

    print("Saved sample predictions with labels to", out_dir)