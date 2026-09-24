import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from utility.startup import device_check, initialize_logger
from utility.paths import CKPT_DIR, PREDS_DIR
from data import generate_loaders, prepare_labels
from model import ECGNet


def evaluate(
        model: nn.Module, 
        loader: DataLoader, 
        criterion: nn.Module | None = None):
    """
    Run a model over an entire split and collect its predictions.

    Puts the model in eval mode and disables gradient tracking, then gathers
    sigmoid probabilities and true labels for every record. Metrics like AUC
    and average precision must be computed on these full arrays, not per batch.

    Args:
        model:     trained ECGNet; its device determines where batches are sent
        loader:    DataLoader for the split to evaluate (val or test)
        criterion: optional loss function; if given, the mean batch loss is returned

    Returns:
        probs:  (N, 5) array of per-class probabilities
        labels: (N, 5) array of 0/1 ground-truth labels, row-aligned with probs
        loss:   mean loss over batches, or None if no criterion was given
    """
    device = next(model.parameters()).device    # pulls the first tensor and reads what device it is on; batches go there too
    model.eval()    # Train -> Eval: BatchNorm1d switches to running averages,  so predictions don't depend on batch composition
    all_probs, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():   # no backward() here, so skip building the gradient graph
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # x:(64, n_leads, 1000), y:(64,5)
            logits = model(x)   # (batch, 5)

            if criterion is not None:
                # If loss function is passed, computes BCE loss and adds it to total_loss
                total_loss += criterion(logits, y).item()  

            # sigmoid turns logits into per-class probabilities; move to CPU so numpy can read them
            all_probs.append(torch.sigmoid(logits).cpu()) 
            all_labels.append(y.cpu())

    # One (N, 5) array per output, one row per record: AUC and AP rank across the whole split
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / len(loader) if criterion is not None else None  # smaller final batch contributes equally

    return probs, labels, loss


def train_one_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        criterion: nn.Module):
    """
    Train the model for one pass over the training set.

    For each batch: clear old gradients, run the forward pass, compute the loss,
    backpropagate, update the weights, then advance the learning-rate schedule.
    The scheduler steps per batch because OneCycleLR is configured with
    steps_per_epoch=len(loader).

    Args:
        model:     ECGNet to train; its device determines where batches are sent
        loader:    training DataLoader (drop_last=True, so every batch is full)
        optimizer: optimizer built over model.parameters()
        scheduler: per-batch learning-rate scheduler (OneCycleLR)
        criterion: loss function, e.g. BCEWithLogitsLoss

    Returns:
        Mean training loss over the epoch's batches.
    """
    device = next(model.parameters()).device    # pulls the first tensor and reads what device it is on; batches go there too

    model.train()   # BatchNorm normalizes with each batch's own stats and updates its running averages    
    train_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # x:(64, n_leads, 1000), y:(64,5)
        optimizer.zero_grad()   # clear gradiants of all optimized parameters before calcuiting gradients for next batch
        logits = model(x)   # forward pass, building the autograd graph
        loss = criterion(logits, y)
        loss.backward()     # backprop: write d(loss)/d(param) into each param.grad
        optimizer.step()    # update weights using those gradients
        scheduler.step()    # then advance the learning rate schedule (must follow optimizer.step())
        train_loss += loss.item()    # .item() detaches, so the graph isn't kept alive
    return train_loss / len(loader)



def run_experiment(
        leads: list[int] | None = None, 
        lead_set_name: str = "full12", 
        seed: int = 42,
        max_lr: float = 3e-3, 
        weight_decay: float = 0.0, 
        n_epochs: int = 10,
        final_eval: bool = False):
    """
    Train and evaluate one configuration of the lead ablation.

    Seeds all randomness, trains an ECGNet on the selected leads, and keeps the
    epoch with the best fold-9 (validation) macro AUC. That checkpoint is then
    evaluated once on the validation set, or on the fold-10 test set when
    final_eval=True. Checkpoint selection always uses validation, so test data
    never influences which model is evaluated.

    Raw predictions are saved to PREDS_DIR for later bootstrap analysis.

    Args:
        leads:         lead indices to train on (0=I ... 11=V6); None means all 12
        lead_set_name: label for this configuration, used in file names and logs
        seed:          seed for weight initialization and shuffle order
        max_lr:        peak learning rate for the OneCycleLR schedule
        weight_decay:  AdamW weight decay
        n_epochs:      training epochs (also sets the OneCycleLR schedule length)
        final_eval:    if True, evaluate on the test fold instead of validation

    Returns:
        dict with run metadata (run_name, lead_set, n_leads, seed, split) and
        metrics: macro AUC, per-class AUC, per-class average precision, and the
        class order they're reported in.
    """
    logger = logging.getLogger(__name__)

    torch.manual_seed(seed)
    np.random.seed(seed)

    split = "test" if final_eval else "val"
    run_name = f"{lead_set_name}_lr{max_lr}_wd{weight_decay}_ep{n_epochs}_seed{seed}_{split}"
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_name}.pt"

    device = device_check()
    loaders = generate_loaders(leads=leads, device=device, workers=0)
    _, _, classes, _ = prepare_labels()

    n_leads = 12 if leads is None else len(leads)
    model = ECGNet(in_channels=n_leads).to(device)  
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)    # Use the AdamW to prevent weight decay accumulation
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=n_epochs,
        steps_per_epoch=len(loaders['train']))  # stepped per batch, not per epoch
    criterion = nn.BCEWithLogitsLoss()  # Sigmoid layer + binary-cross entropy

    best_auc = 0.0
    epoch_range = tqdm(range(n_epochs), colour="cyan",
                   desc=f"{lead_set_name} s{seed}", leave=False)
    
    for epoch in epoch_range:      
        train_loss = train_one_epoch(model, loaders['train'], optimizer, scheduler, criterion)
        val_probs, val_labels, val_loss = evaluate(model, loaders['val'], criterion)
        val_auc = roc_auc_score(val_labels, val_probs, average='macro')

        # tqdm.write(f"Epoch {epoch} has finished.")
        logger.info(f"{run_name} epoch {epoch}: train {train_loss:.4f} | "
                    f"val {val_loss:.4f} | auc {val_auc:.4f}")
        if val_auc > best_auc:  # limited to validation split; test never influences which epoch is kept 
            best_auc = val_auc
            torch.save(model.state_dict(), ckpt_path)

    # evaluate the best epoch, not whatever the last epoch happened to be
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    eval_loader = loaders['test'] if final_eval else loaders['val']
    probs, labels, _ = evaluate(model, eval_loader)

    macro = roc_auc_score(labels, probs, average='macro')
    per_class = roc_auc_score(labels, probs, average=None)
    ap = average_precision_score(labels, probs, average=None)   

    PREDS_DIR.mkdir(exist_ok=True)
    np.savez(PREDS_DIR / f"{run_name}.npz", probs=probs, labels=labels)   # raw predictions for
                                                                          # patient-level bootstrap CIs

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