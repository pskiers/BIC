import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import timm
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataset_utils import LogoAugmentedDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train Logo Detector with WandB")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "mobilenet", "vit"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--fixed_logo", action="store_true", help="Use fixed logo size")
    parser.add_argument("--no_streaming", action="store_true", help="Download dataset instead of streaming")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_steps", type=int, default=4000, help="Steps per train epoch")
    parser.add_argument("--val_steps", type=int, default=150, help="Steps per val epoch")
    parser.add_argument("--project_name", type=str, default="logo-detection")
    return parser.parse_args()


def get_model_name(args):
    if args.model == "resnet18":
        return "resnet18"
    elif args.model == "mobilenet":
        return "mobilenetv2_100"
    elif args.model == "vit":
        return "vit_base_patch16_224"
    raise ValueError("Unknown model")


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return acc, precision, recall, f1


def train_one_epoch(model, loader, optimizer, criterion, device, steps, epoch_idx):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    data_iter = iter(loader)

    for step in range(steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if step % 50 == 0:
            wandb.log({"train_batch_loss": loss.item()})

    avg_loss = running_loss / max(1, step)
    acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1


def validate(model, loader, criterion, device, steps):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    data_iter = iter(loader)

    with torch.no_grad():
        for step in range(steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(1, step)
    acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1


def main():
    args = get_args()

    run_name = f"{args.model}_pre-{int(args.pretrained)}_fixed-{int(args.fixed_logo)}_lr-{args.lr}"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Run: {run_name}")

    pre_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip()])
    post_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 4. Initialize Datasets (Streaming or Downloaded)
    is_streaming = not args.no_streaming

    print("--- Initializing Training Set ---")
    train_dataset = LogoAugmentedDataset(
        num_logos=20,
        split="train",
        pre_transform=pre_transform,
        post_transform=post_transform,
        fixed_logo_size=args.fixed_logo,
        streaming=is_streaming,
    )

    print("--- Initializing Validation Set ---")
    val_dataset = LogoAugmentedDataset(
        num_logos=20,
        split="validation",
        pre_transform=pre_transform,
        post_transform=post_transform,
        fixed_logo_size=args.fixed_logo,
        streaming=is_streaming,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 5. Model Setup
    model_name = get_model_name(args)
    print(f"Loading Model: {model_name}")
    model = timm.create_model(model_name, pretrained=args.pretrained, num_classes=20)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Training Loop
    for epoch in range(args.epochs):
        # TRAIN
        print(f"Epoch {epoch+1}: Training...")
        t_loss, t_acc, t_prec, t_rec, t_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.train_steps, epoch
        )

        # VAL
        print(f"Epoch {epoch+1}: Validating...")
        v_loss, v_acc, v_prec, v_rec, v_f1 = validate(model, val_loader, criterion, device, args.val_steps)

        print(f"Results Epoch {epoch+1}: Train Acc: {t_acc:.3f} | Val Acc: {v_acc:.3f}")

        # Log to WandB
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": t_loss,
                "train/acc": t_acc,
                "train/prec": t_prec,
                "train/rec": t_rec,
                "train/f1": t_f1,
                "val/loss": v_loss,
                "val/acc": v_acc,
                "val/prec": v_prec,
                "val/rec": v_rec,
                "val/f1": v_f1,
            }
        )

        torch.save(model.state_dict(), f"{run_name}_ep{epoch+1}.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
