import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import Config
from models import ImageEncoder, Stage1Model, Stage2Model
from dataset import Stage2Dataset
from utils import save_checkpoint, Logger


def train_stage2():
    Config.create_dirs()
    device = Config.device
    print(f"Using device: {device}")

    # 加载第一阶段图像编码器
    img_encoder = ImageEncoder(out_dim=Config.img_feat_dim, pretrained=True).to(device)
    stage1_model = Stage1Model(img_encoder, num_maturity=Config.maturity_classes,
                               num_freshness=Config.freshness_classes).to(device)
    checkpoint = torch.load(os.path.join(Config.stage1_save_dir, 'best_model.pth'), map_location=device)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_img_encoder = stage1_model.img_encoder

    # 冻结部分层：只微调最后几层
    for name, param in pretrained_img_encoder.named_parameters():
        if 'layer4' in name or 'projection' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = Stage2Model(pretrained_img_encoder, Config).to(device)

    train_dataset = Stage2Dataset(Config.data_root, Config, is_train=True)
    val_dataset = Stage2Dataset(Config.data_root, Config, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=Config.stage2_batch_size, shuffle=True,
                              num_workers=Config.stage2_num_workers, collate_fn=Stage2Dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.stage2_batch_size, shuffle=False,
                            num_workers=Config.stage2_num_workers, collate_fn=Stage2Dataset.collate_fn)

    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.stage2_lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    start_epoch = 0

    # 断点续训：检测 resume checkpoint
    resume_path = os.path.join(Config.stage2_save_dir, 'resume.pth')
    is_resume = os.path.exists(resume_path)
    if is_resume:
        print(f"Found resume checkpoint: {resume_path}, resuming training...")
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        best_val_loss = resume_ckpt['best_val_loss']
        start_epoch = resume_ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    logger = Logger(os.path.join(Config.log_dir, 'stage2_log.txt'), append=is_resume)

    for epoch in range(start_epoch, Config.stage2_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{Config.stage2_epochs}")
        for batch in pbar:
            img_seq = batch['img_seq'].to(device)
            et_seq = batch['et_seq'].to(device)
            enz_seq = batch['enz_seq'].to(device)
            wl_seq = batch['wl_seq'].to(device)
            maturity = batch['maturity'].to(device)
            freshness = batch['freshness'].to(device)
            storage_cond = batch['storage_cond'].to(device)
            fcr_gt = batch['fcr'].to(device)
            moisture_gt = batch['moisture'].to(device)
            hardness_gt = batch['hardness'].to(device)

            optimizer.zero_grad()
            fcr_pred, moisture_pred, hardness_pred = model(img_seq, et_seq, enz_seq, wl_seq, maturity, freshness,
                                                           storage_cond)
            loss_fcr = criterion_reg(fcr_pred.squeeze(), fcr_gt)
            loss_moist = criterion_reg(moisture_pred.squeeze(), moisture_gt)
            loss_hard = criterion_reg(hardness_pred.squeeze(), hardness_gt)
            loss = Config.loss_weights_stage2['fcr'] * loss_fcr + \
                   Config.loss_weights_stage2['moisture'] * loss_moist + \
                   Config.loss_weights_stage2['hardness'] * loss_hard
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # 验证
        model.eval()
        val_loss = 0.0
        all_fcr_gt, all_fcr_pred = [], []
        all_moist_gt, all_moist_pred = [], []
        all_hard_gt, all_hard_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                img_seq = batch['img_seq'].to(device)
                et_seq = batch['et_seq'].to(device)
                enz_seq = batch['enz_seq'].to(device)
                wl_seq = batch['wl_seq'].to(device)
                maturity = batch['maturity'].to(device)
                freshness = batch['freshness'].to(device)
                storage_cond = batch['storage_cond'].to(device)
                fcr_gt = batch['fcr'].to(device)
                moisture_gt = batch['moisture'].to(device)
                hardness_gt = batch['hardness'].to(device)
                fcr_pred, moisture_pred, hardness_pred = model(img_seq, et_seq, enz_seq, wl_seq, maturity, freshness,
                                                               storage_cond)
                loss = (criterion_reg(fcr_pred.squeeze(), fcr_gt) +
                        criterion_reg(moisture_pred.squeeze(), moisture_gt) +
                        criterion_reg(hardness_pred.squeeze(), hardness_gt))
                val_loss += loss.item()

                all_fcr_gt.extend(fcr_gt.cpu().tolist())
                all_fcr_pred.extend(fcr_pred.squeeze().cpu().tolist())
                all_moist_gt.extend(moisture_gt.cpu().tolist())
                all_moist_pred.extend(moisture_pred.squeeze().cpu().tolist())
                all_hard_gt.extend(hardness_gt.cpu().tolist())
                all_hard_pred.extend(hardness_pred.squeeze().cpu().tolist())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 逐样本预测结果
        pred_lines = ["  Sample | FCR(GT/Pred) | Moisture(GT/Pred) | Hardness(GT/Pred)"]
        for i in range(len(all_fcr_gt)):
            pred_lines.append(
                f"  [{i+1:>3}]  | {all_fcr_gt[i]:6.4f} / {all_fcr_pred[i]:6.4f}"
                f" | {all_moist_gt[i]:6.4f} / {all_moist_pred[i]:6.4f}"
                f" | {all_hard_gt[i]:6.4f} / {all_hard_pred[i]:6.4f}"
            )
        pred_detail = "\n".join(pred_lines)

        log_msg = (f"Epoch {epoch + 1}: Train Loss={train_loss / len(train_loader):.4f}, "
                   f"Val Loss={val_loss:.4f}\n{pred_detail}")
        print(log_msg)
        logger.log(log_msg)

        # 每个 epoch 保存续训 checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, resume_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(Config.stage2_save_dir, 'best_model.pth'))

    # 训练完成后删除续训 checkpoint，避免下次误用
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("Stage2 training completed.")


if __name__ == '__main__':
    train_stage2()