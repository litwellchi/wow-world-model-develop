import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.dataset import RLBenchFLowFeatureDataset
import timm
from torchvision import transforms


def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_model_class(model_name):
    if model_name == 'dinov2_flow':
        from models.idm_models import Dino3DFlowIDM
        return Dino3DFlowIDM
    else:
        raise ValueError(f"Unknown model type: {model_name}")



def main(args):
    # Load DINO + CoTracker
    model = get_model_class(args.model)(output_dim=args.output_dim)
    
    dino = model.encoder
    dino.head = model.encoder.head 
    dino_transform = model.transforms
    flow_tracker = model.flow_tracker

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    # Dataset
    dataset = RLBenchFLowFeatureDataset(
        rootdir=args.data_path,
        dino_model=dino,
        cotracker_model=flow_tracker,
        dino_transform=dino_transform
    )

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader = cycle(train_loader)

    # Model

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))

    # Resume
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"[INFO] Resumed from step {start_step}")

    # Training loop
    with tqdm(range(start_step, args.total_steps), total=args.total_steps - start_step) as pbar:
        for step in pbar:
            total_loss = 0
            for _ in range(args.gradient_accumulation):
                img_feat, flow_feat, action, text_id = next(train_loader)
                img_feat, flow_feat, action, text_id = img_feat.cuda(), flow_feat.cuda(), action.cuda(), text_id.cuda()

                pred = model(img_feat, flow_feat, text_id)
                loss = F.mse_loss(pred, action)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_train_loss = total_loss / args.gradient_accumulation
            pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})
            writer.add_scalar("Loss/Train", avg_train_loss, step)

            if step % 200 == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        img_feat, flow_feat, action, text_id = batch
                        img_feat, flow_feat, action, text_id = img_feat.cuda(), flow_feat.cuda(), action.cuda(), text_id.cuda()
                        pred = model(img_feat, flow_feat, text_id)
                        val_loss += F.mse_loss(pred, action).item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"[Step {step}] Val Loss: {avg_val_loss:.4f}")
                writer.add_scalar("Loss/Val", avg_val_loss, step)
                model.train()

            if step % args.save_every == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step
                }, os.path.join(args.save_path, f"checkpoint_{step}.pth"))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to RLBench dataset')
    parser.add_argument('--save_path', default='./checkpoints', help='Directory to save checkpoints and logs')
    parser.add_argument('--resume_from', default='', help='Path to resume checkpoint')
    parser.add_argument('--total_steps', type=int, default=20000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=5, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--output_dim', type=int, default=7, help='Action output dimension')
    parser.add_argument('--model', type=str, default='dinov2', choices=['dinov2_flow','dinov2_flow_transformer'],
                    help='Choose model type: dinov2 or resnet')
    args = parser.parse_args()

    main(args)