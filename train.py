import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from utility.startup import device_check, initialize_logger
from utility.paths import CKPT_DIR
from data import generate_loaders, prepare_labels
from model import ECGNet


def evaluate(model, loader, criterion=None):
    device = next(model.parameters()).device

    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)

            if criterion is not None:
                total_loss += criterion(logits, y).item()

            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / len(loader) if criterion is not None else None

    return probs, labels, loss


def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    device = next(model.parameters()).device

    model.train()
    train_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    return train_loss / len(loader)



def run_experiment(leads=None, lead_set_name="full12", seed=42,
                   max_lr=3e-3, weight_decay=0.0, n_epochs=10,
                   final_eval=False):
    logger = logging.getLogger(__name__)

    torch.manual_seed(seed)
    np.random.seed(seed)

    run_name = f"{lead_set_name}_lr{max_lr}_wd{weight_decay}_ep{n_epochs}_seed{seed}"
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_name}.pt"

    device = device_check()
    loaders = generate_loaders(leads=leads, device=device, workers=0)
    _, _, classes, _ = prepare_labels()

    n_leads = 12 if leads is None else len(leads)
    model = ECGNet(in_channels=n_leads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=n_epochs,
        steps_per_epoch=len(loaders['train']))
    criterion = nn.BCEWithLogitsLoss()

    print("\n\nBeginning training...\n\n")

    best_auc = 0.0

    for epoch in tqdm(range(n_epochs), 
                      desc="Training", colour=("#824bf8")
                      ):
        train_loss = train_one_epoch(model, loaders['train'], optimizer, scheduler, criterion)
        val_probs, val_labels, val_loss = evaluate(model, loaders['val'], criterion)
        val_auc = roc_auc_score(val_labels, val_probs, average='macro')

        tqdm.write(f"Epoch {epoch} has finished.")
        logger.info(f"{run_name} epoch {epoch}: train {train_loss:.4f} | "
                    f"val {val_loss:.4f} | auc {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), ckpt_path)
    print("\n\nDone with training!\n\n")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    eval_loader = loaders['test'] if final_eval else loaders['val']
    probs, labels, _ = evaluate(model, eval_loader)

    macro = roc_auc_score(labels, probs, average='macro')
    per_class = roc_auc_score(labels, probs, average=None)
    ap = average_precision_score(labels, probs, average=None)

    return {
        "run_name": run_name,
        "lead_set": lead_set_name,
        "n_leads": n_leads,
        "seed": seed,
        "split": "test" if final_eval else "val",
        "macro": macro,
        "per_class": dict(zip(classes, per_class)),
        "per_class_ap": dict(zip(classes, ap)),
        "classes": classes,
    }

if __name__ == "__main__":
    initialize_logger()
    print(run_experiment())