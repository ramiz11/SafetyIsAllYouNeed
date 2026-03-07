
import argparse
import os
import pickle as pkl
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from configs import preprocessing_config as pc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajDataset(Dataset):
    def __init__(self, trajs):
        self.inputs = []
        self.labels = []
        for df in trajs:
            seq = df['poi_id'].astype(int).tolist()
            if len(seq) < 2:
                continue
            self.inputs.append(seq[:-1])
            self.labels.append(seq[-1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class LSTMGRU(nn.Module):
    def __init__(self, num_pois, emb_dim=128, hidden_dim=256, model_type='lstm'):
        super().__init__()
        self.embed = nn.Embedding(num_pois, emb_dim)
        rnn_cls = nn.LSTM if model_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_pois)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        last = out[:, -1, :]
        return self.fc(last)


def acc_at_k(logits, labels, k):
    topk = torch.topk(logits, k=k, dim=1).indices
    match = (topk == labels.view(-1, 1)).any(dim=1)
    return match.float().mean().item()


def mrr_at_k(logits, labels, k):
    topk = torch.topk(logits, k=k, dim=1).indices
    rr = []
    for i in range(labels.size(0)):
        lab = labels[i].item()
        preds = topk[i].tolist()
        rank = preds.index(lab) + 1 if lab in preds else None
        rr.append(1.0 / rank if rank else 0.0)
    return float(np.mean(rr)) if rr else 0.0


def evaluate(model, loader, device, k_list=(1,3,5)):
    model.eval()
    total = 0
    accs = {k: 0.0 for k in k_list}
    mrrs = {k: 0.0 for k in k_list}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total += 1
            for k in k_list:
                accs[k] += acc_at_k(logits, y, k)
                mrrs[k] += mrr_at_k(logits, y, k)
    # average over batches
    for k in k_list:
        accs[k] /= max(total, 1)
        mrrs[k] /= max(total, 1)
    return accs, mrrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','eval','both'], default='both')
    parser.add_argument('--model', choices=['lstm','gru'], default='lstm')
    parser.add_argument('--dataset', default='NYC')
    parser.add_argument('--traj-len', type=int, default=20)
    parser.add_argument('--crime-radius', type=int, default=500)
    parser.add_argument('--crime-time-weeks', type=int, default=4)
    parser.add_argument('--base-dir', default='/absolute/path/to/SafetyIsAllYouNeed')
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    pc.update_config(args.dataset, args.traj_len, args.crime_radius, args.crime_time_weeks, base_dir=args.base_dir)

    # load trajectories
    with open(pc.TRAIN_TRAJECTORIES_PKL_PATH, 'rb') as f:
        train_trajs = pkl.load(f)
    with open(pc.VALIDATION_TRAJECTORIES_PKL_PATH, 'rb') as f:
        val_trajs = pkl.load(f)
    with open(pc.TEST_TRAJECTORIES_PKL_PATH, 'rb') as f:
        test_trajs = pkl.load(f)

    # num pois
    all_pois = set()
    for df in train_trajs + val_trajs + test_trajs:
        all_pois.update(df['poi_id'].astype(int).tolist())
    num_pois = max(all_pois) + 1 if all_pois else 1

    train_ds = TrajDataset(train_trajs)
    val_ds = TrajDataset(val_trajs)
    test_ds = TrajDataset(test_trajs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMGRU(num_pois, args.emb_dim, args.hidden_dim, model_type=args.model).to(device)

    save_dir = os.path.join(args.base_dir, 'baselines', 'lstm_gru', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{args.model}_{args.dataset}_len{args.traj_len}.pt")

    if args.mode in ('train','both'):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val = float('inf')
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += loss_fn(model(x), y).item()
            val_loss /= max(len(val_loader), 1)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {epoch} | val_loss={val_loss:.4f}")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    if args.mode in ('eval','both'):
        accs, mrrs = evaluate(model, test_loader, device)
        print('=== TEST METRICS ===')
        for k, v in accs.items():
            print(f"Acc@{k}: {v:.4f}")
        for k, v in mrrs.items():
            print(f"MRR@{k}: {v:.4f}")


if __name__ == '__main__':
    main()
