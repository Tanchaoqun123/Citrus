import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from config import Config
from models import ImageEncoder, Stage1Model
from dataset import Stage1Dataset
from utils import save_checkpoint, Logger


def train_stage1():
    Config.create_dirs()
    device = Config.device
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Stage1Dataset(Config.data_root, transform=train_transform, is_train=True)
    val_dataset = Stage1Dataset(Config.data_root, transform=val_transform, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=Config.stage1_batch_size, shuffle=True,
                              num_workers=Config.stage1_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.stage1_batch_size, shuffle=False,
                            num_workers=Config.stage1_num_workers)

    img_encoder = ImageEncoder(out_dim=Config.img_feat_dim, pretrained=True).to(device)
    model = Stage1Model(img_encoder, num_maturity=Config.maturity_classes, num_freshness=Config.freshness_classes).to(
        device)

    criterion_maturity = nn.CrossEntropyLoss()
    criterion_freshness = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.stage1_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.stage1_epochs)

    best_val_loss = float('inf')
    best_acc_m = 0.0
    best_acc_f = 0.0
    logger = Logger(os.path.join(Config.log_dir, 'stage1_log.txt'))

    for epoch in range(Config.stage1_epochs):
        model.train()
        train_loss = 0.0
        train_correct_m = train_correct_f = train_total_m = train_total_f = 0
        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}/{Config.stage1_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            maturity = batch['maturity'].to(device)
            freshness = batch['freshness'].to(device)

            optimizer.zero_grad()
            out_m, out_f, _ = model(images)
            mask_m = maturity >= 0
            mask_f = freshness >= 0
            loss_m = criterion_maturity(out_m[mask_m], maturity[mask_m]) if mask_m.any() else torch.tensor(0.0, device=device)
            loss_f = criterion_freshness(out_f[mask_f], freshness[mask_f]) if mask_f.any() else torch.tensor(0.0, device=device)
            loss = Config.loss_weights_stage1['maturity'] * loss_m + \
                   Config.loss_weights_stage1['freshness'] * loss_f
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if mask_m.any():
                train_correct_m += (out_m[mask_m].argmax(dim=1) == maturity[mask_m]).sum().item()
                train_total_m += mask_m.sum().item()
            if mask_f.any():
                train_correct_f += (out_f[mask_f].argmax(dim=1) == freshness[mask_f]).sum().item()
                train_total_f += mask_f.sum().item()

            pbar.set_postfix({'loss': loss.item()})

        train_acc_m = train_correct_m / train_total_m if train_total_m > 0 else 0.0
        train_acc_f = train_correct_f / train_total_f if train_total_f > 0 else 0.0

        # 验证
        model.eval()
        val_loss = 0.0
        correct_m = correct_f = total_m = total_f = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                maturity = batch['maturity'].to(device)
                freshness = batch['freshness'].to(device)
                out_m, out_f, _ = model(images)
                mask_m = maturity >= 0
                mask_f = freshness >= 0
                loss_m = criterion_maturity(out_m[mask_m], maturity[mask_m]) if mask_m.any() else torch.tensor(0.0, device=device)
                loss_f = criterion_freshness(out_f[mask_f], freshness[mask_f]) if mask_f.any() else torch.tensor(0.0, device=device)
                loss = loss_m + loss_f
                val_loss += loss.item()

                if mask_m.any():
                    pred_m = out_m[mask_m].argmax(dim=1)
                    correct_m += (pred_m == maturity[mask_m]).sum().item()
                    total_m += mask_m.sum().item()
                if mask_f.any():
                    pred_f = out_f[mask_f].argmax(dim=1)
                    correct_f += (pred_f == freshness[mask_f]).sum().item()
                    total_f += mask_f.sum().item()

        val_loss /= len(val_loader)
        val_acc_m = correct_m / total_m if total_m > 0 else 0.0
        val_acc_f = correct_f / total_f if total_f > 0 else 0.0
        scheduler.step()

        log_msg = (f"Epoch {epoch + 1}: "
                   f"Train Loss={train_loss / len(train_loader):.4f}, "
                   f"Train Maturity Acc={train_acc_m:.4f}, Train Freshness Acc={train_acc_f:.4f} | "
                   f"Val Loss={val_loss:.4f}, "
                   f"Val Maturity Acc={val_acc_m:.4f}, Val Freshness Acc={val_acc_f:.4f}")
        print(log_msg)
        logger.log(log_msg)

        if val_acc_m > best_acc_m:
            best_acc_m = val_acc_m
        if val_acc_f > best_acc_f:
            best_acc_f = val_acc_f

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'img_encoder_state_dict': model.img_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(Config.stage1_save_dir, 'best_model.pth'))

    best_msg = f"Best Maturity Acc={best_acc_m:.4f}, Best Freshness Acc={best_acc_f:.4f}"
    print(best_msg)
    logger.log(best_msg)
    print("Stage1 training completed.")


if __name__ == '__main__':
    train_stage1()