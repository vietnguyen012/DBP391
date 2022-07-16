import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class MarkdownModel1(nn.Module):
    def __init__(self):
        super(MarkdownModel1, self).__init__()
        self.distill_bert = AutoModel.from_pretrained("./checkpoint/models/checkpoint-18000")
        self.top = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x)
        return x

class MarkdownModel2(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel2, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.top(x)
        return x