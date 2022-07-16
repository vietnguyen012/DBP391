import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 8  # 666
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
nd = torch.cuda.device_count()

import torch.distributed as dist
from contextlib import contextmanager

data_dir = Path('./')

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./data/val.csv")
parser.add_argument('--train_path', type=str, default="./data/train.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--do_eval', action='store_true')

args = parser.parse_args()

def read_data(data,device):
    return tuple(d.to(device) for d in data[:-1]), data[-1].to(device)


def validate(model, val_loader,device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data,device)

            # with torch.cuda.amp.autocast():
            pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs,local_rank=None,world_size=None,device=None):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    # scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data,device)

            # with torch.cuda.amp.autocast():
            pred = model(*inputs)
            loss = criterion(pred, target)
            loss *= world_size
            loss.backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
        if local_rank in [-1,0]:
            torch.save(model.state_dict(), f"./outputs/model_pp3_{avg_loss}_{e}.bin")

    return model

data_dir = Path('.')

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
train_df = pd.read_csv(args.train_path)
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

order_df = pd.read_csv("./train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

if not args.do_eval:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK'))
    device = torch.device(f'cuda:{local_rank}')
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', -1))
    torch.cuda.set_device(device)
    global_rank = dist.get_rank()
    if local_rank in [-1,0]:
        if not os.path.exists("./outputs"):
            os.mkdir("./outputs")

    train_ds = MarkdownDataset2(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                               total_max_len=args.total_max_len, fts=train_fts)
    val_ds = MarkdownDataset2(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                             total_max_len=args.total_max_len, fts=val_fts)
    with torch_distributed_zero_first(rank):
        sampler = DistributedSampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size// world_size, sampler=sampler, num_workers=args.n_workers,
                                  pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32,
                            pin_memory=False, drop_last=False)
else:
    train_ds = MarkdownDataset2(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                               total_max_len=args.total_max_len, fts=train_fts)
    val_ds = MarkdownDataset2(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                             total_max_len=args.total_max_len, fts=val_fts)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            pin_memory=False, drop_last=False)

if not args.do_eval:
    model = MarkdownModel2(args.model_name_or_path)
    # model.load_state_dict(torch.load("./outputs/model.bin"))
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = train(model, train_loader, val_loader, epochs=args.epochs,local_rank=local_rank,world_size=world_size,device=device)
else:
    model = MarkdownModel2(args.model_name_or_path)
    device='cuda'
    model = model.to(device)
    checkpoint = torch.load("./outputs/model_pp3_0.1448_14.bin")
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    y_val, y_pred = validate(model, val_loader,device)
    val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    score,low_score_item_dict = kendall_tau(df_orders.loc[y_dummy.index], y_dummy,return_low_example=True)
    print('score : ', score)
    print("low score item:",low_score_item_dict)
    low_score_item_dict["score_train_validation"] = score
    import pickle
    with open('./outputs/low_example.pickle', 'wb') as handle:
        pickle.dump(low_score_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)