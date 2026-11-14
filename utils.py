import copy, time
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def kd_loss(student_logits, teacher_logits, labels, T=3.0, lam=0.5):
    ce = F.cross_entropy(student_logits, labels)

    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction="batchmean")

    return (1 - lam) * ce + lam * T**2 * kl


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    loss, top1, top5, n = 0.0, 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.topk(5, dim=1).indices 

        loss += F.cross_entropy(logits, y, reduction="sum").item()
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y[:, None]).any(dim=1).sum().item()
        n += len(y)

    return {"loss": loss / n, "top1": top1 / n, "top5": top5 / n}


def train(model, loss_fn, optimizer, train_loader, val_loader, epochs, device="cpu", save_path=None):

    model.to(device)
    history = defaultdict(list)
    best_top1, best_state, updates, train_time = -1.0, None, 0, 0.0

    for epoch in range(epochs):
        model.train()
        running, correct, n = defaultdict(float), 0, 0
        start = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs - 1}"):
            x, y = x.to(device), y.to(device)

            loss, logits, extras = loss_fn(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1

            running["loss"] += loss.item() * len(y)
            for k, v in extras.items():
                running[k] += float(v) * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)

        train_time += time.time() - start
        val = evaluate(model, val_loader, device)

        for k, v in running.items():
            history[f"train_{k}"].append(v / n)
        history["train_top1"].append(correct / n)
        for k, v in val.items():
            history[f"val_{k}"].append(v)

        print(f"Epoch {epoch}/{epochs - 1} | train loss {history['train_loss'][-1]:.4f} "
              f"| val loss {val['loss']:.4f} top1 {val['top1']:.4f} top5 {val['top5']:.4f}")

        if val["top1"] > best_top1:
            best_top1, best_state = val["top1"], copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch

    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)

    history["updates"], history["train_time_s"] = updates, train_time
    return dict(history)
