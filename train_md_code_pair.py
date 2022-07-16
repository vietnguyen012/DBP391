
import numpy as np
from tqdm import tqdm
import sys, os
import random
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
seed = 8  # 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
nd = torch.cuda.device_count()
from metrics import *
from dataset import *
import torch.distributed as dist
try:
    from transformers import DistilBertModel, DistilBertTokenizer
except:
    from transformers import DistilBertModel, DistilBertTokenizer

try:
    from transformers import DistilBertModel, DistilBertTokenizer
except:
    from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from model import *
import argparse
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--val_path', type=str, default="./data/val.csv")
parser.add_argument('--train_path', type=str, default="./data/train.csv")
parser.add_argument('--do_eval', action="store_true")

args = parser.parse_args()

data_dir = Path('./')

do_train = not args.do_eval
do_eval = args.do_eval

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


def generate_triplet(df, mode='train'):
    triplets = []
    ids = df.id.unique()
    random_drop = np.random.random(size=10000) > 0.9
    count = 0

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']

        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values

        for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
            labels = np.array([(r == (rank + 1)) for r in df_tmp_code_rank]).astype('int')

            for cid, label in zip(df_tmp_code_cell_id, labels):
                count += 1
                if label == 1:
                    triplets.append([cell_id, cid, label])
                    # triplets.append( [cid, cell_id, label] )
                elif mode == 'test':
                    triplets.append([cell_id, cid, label])
                    # triplets.append( [cid, cell_id, label] )
                elif random_drop[count % 10000]:
                    triplets.append([cell_id, cid, label])
                    # triplets.append( [cid, cell_id, label] )

    return triplets

def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 5e-5
    elif epoch < 5:
        lr = 5e-5
    else:
        lr = 5e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr


def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-8)  # 1e-08)
    return optimizer

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader, mode='train'):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            if mode == 'test':
                labels.append(target.detach().cpu().numpy().ravel())
    if mode == 'test':
        return np.concatenate(preds)
    else:
        return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, epochs, Type='markdown',local_rank=None,world_size=None):
    np.random.seed(0)

    optimizer = get_optimizer(model)

    mixed_precision = True
    mixed_precision = True
    try:
        from apex import amp
    except:
        mixed_precision = False

    #model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=1)

    criterion = torch.nn.L1Loss()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)

        lr = adjust_lr(optimizer, e)

        loss_list = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            optimizer.zero_grad()
            pred = model(inputs[0], inputs[1])
            loss = criterion(pred, target)
            loss *= world_size
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().item())
            avg_loss = np.round(np.mean(loss_list), 4)
        if local_rank in [0,-1]:
            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {lr}")

            output_model_file = f"./pp2_output/my_own_model_pp2_{avg_loss}_{e}.bin"
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)

    return model

data_dir = Path('.')

train_df = pd.read_csv(args.train_path)
val_df = pd.read_csv(args.val_path)

order_df = pd.read_csv("./train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

if do_train:
    dict_cellid_source = dict(zip(train_df['cell_id'].values, train_df['source'].values))
    triplets = generate_triplet(train_df)
else:
    dict_cellid_source = dict(zip(val_df['cell_id'].values, val_df['source'].values))
    triplets = generate_triplet(val_df,mode='test')

MAX_LEN = 128

if do_train:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK'))
    device = torch.device(f'cuda:{local_rank}')
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', -1))
    torch.cuda.set_device(device)
    global_rank = dist.get_rank()

    with torch_distributed_zero_first(rank):
        BS = 500
        NW = 8
        train_ds = MarkdownDataset1(triplets, max_len=MAX_LEN,dict_cellid_source=dict_cellid_source,mode='test')
        sampler = DistributedSampler(train_ds)
        train_loader = DataLoader(dataset=train_ds, batch_size=BS // world_size, sampler=sampler, num_workers=NW,
                              pin_memory=False)
        print("num of examples:",len(train_loader))
elif do_eval:
    train_ds = MarkdownDataset1(triplets, max_len=MAX_LEN,dict_cellid_source=dict_cellid_source, mode='test')
    BS = 128
    NW = 8
    val_loader = DataLoader(train_ds, batch_size=BS, shuffle=False, num_workers=NW,
                              pin_memory=False, drop_last=False)
    print("num of examples:", len(val_loader))

if do_train:
    model = MarkdownModel1().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # model = model.cuda()
    print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
        os.environ.get('WORLD_SIZE'),
        os.environ.get('RANK'),
        os.environ.get('LOCAL_RANK'),
        os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    model = train(model, train_loader, epochs=9, Type='markdown',local_rank=local_rank,world_size=world_size)
elif do_eval:
    model = MarkdownModel1()
    model = model.cuda()
    model.load_state_dict(torch.load('./pp2_output/my_own_model_pp2_0.2747_8.bin'))
    y_val = validate(model, val_loader, mode='test')
    preds_copy = y_val
    pred_vals = []
    count = 0
    for id, df_tmp in tqdm(val_df.groupby('id')):
        df_tmp_mark = df_tmp[df_tmp['cell_type'] == 'markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type'] != 'markdown']
        df_tmp_code_rank = df_tmp_code['rank'].rank().values
        N_code = len(df_tmp_code_rank)
        N_mark = len(df_tmp_mark)

        preds_tmp = preds_copy[count:count + N_mark * N_code]

        count += N_mark * N_code

        for i in range(N_mark):
            pred = preds_tmp[i * N_code:i * N_code + N_code]

            softmax = np.exp((pred - np.mean(pred)) * 20) / np.sum(np.exp((pred - np.mean(pred)) * 20))

            rank = np.sum(softmax * df_tmp_code_rank)
            pred_vals.append(rank)

    del model
    del triplets
    del dict_cellid_source
    gc.collect()
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = pred_vals
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    score,low_score_item_dict = kendall_tau(df_orders.loc[y_dummy.index], y_dummy,return_low_example=True)
    print('score : ', score)
    print("low score item:",low_score_item_dict)
    low_score_item_dict["score_train_validation"] = score
    import pickle
    with open('./pp2_output/low_example.pickle', 'wb') as handle:
        pickle.dump(low_score_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
