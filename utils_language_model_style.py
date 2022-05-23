import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import random


class DatasetWithMask_old(Dataset):
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        seq_raw_torch_list = []
        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []

        for i in range(self.seq_num):
            seq_raw_torch = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            seq_raw_torch_list.append(seq_raw_torch)
            seq_torch = seq_raw_torch[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi['<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)
        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_torch_pad_all = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:
            clicked_items = set(list(seq_torch_pad_all.numpy()))
            target_id_neg_torch_list = []
            for h in range(seq_torch_pad.shape[0]):

                target_id_neg = []

                for _ in range(self.num_neg):
                    target_id_neg_i = torch.randint(low=2, high=self.vocab_size+2, size=[]).item()
                    while target_id_neg_i in clicked_items:
                        target_id_neg_i = torch.randint(low=2, high=self.vocab_size+2, size=[]).item()
                    target_id_neg.append(target_id_neg_i)
                target_id_neg_torch_h = torch.tensor(target_id_neg, dtype=torch.long)
                target_id_neg_torch_list.append(target_id_neg_torch_h)
            target_id_neg_torch = torch.stack(target_id_neg_torch_list, dim=0)

        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
        #
        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


class DatasetWithMask(Dataset):
    """
    simple neg sampling
    """
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []

        for i in range(self.seq_num):
            seq_torch = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            seq_torch = seq_torch[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi['<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)
        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_torch_pad_all = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:
            # clicked_items = set(list(seq_torch_pad_all.numpy()))
            # target_id_neg_torch_list = []
            # for h in range(seq_torch_pad.shape[0]):
            #
            #     target_id_neg = []
            #
            #     for _ in range(self.num_neg):
            #         target_id_neg_i = torch.randint(low=2, high=self.vocab_size, size=[]).item()
            #         while target_id_neg_i in clicked_items:
            #             target_id_neg_i = torch.randint(low=2, high=self.vocab_size, size=[]).item()
            #         target_id_neg.append(target_id_neg_i)
            #     target_id_neg_torch_h = torch.tensor(target_id_neg, dtype=torch.long)
            #     target_id_neg_torch_list.append(target_id_neg_torch_h)
            # target_id_neg_torch = torch.stack(target_id_neg_torch_list, dim=0)
            target_id_neg_torch = torch.randint(low=2, high=self.vocab_size+2, size=[target_id_torch.shape[0],
                                                                                   self.num_neg])
        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
        #
        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


class DatasetWithMaskEval(Dataset):
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []

        for i in range(self.seq_num):
            seq_torch = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            seq_torch = seq_torch[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi[
                '<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)
        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_torch_pad_all = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:

            target_id_neg = []

            clicked_items = set(list(seq_torch_pad_all.numpy()))

            for _ in range(self.num_neg):
                target_id_neg_i = torch.randint(low=2, high=self.vocab_size+2, size=[]).item()
                while target_id_neg_i in clicked_items:
                    target_id_neg_i = torch.randint(low=2, high=self.vocab_size+2, size=[]).item()
                target_id_neg.append(target_id_neg_i)
            target_id_neg_torch = torch.tensor(target_id_neg, dtype=torch.long)

        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
        #
        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


class DatasetWithMaskEvalAll(Dataset):
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        self.seq_raw_torch_list = []
        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []

        for i in range(self.seq_num):
            seq_torch_raw = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            self.seq_raw_torch_list.append(seq_torch_raw)
            seq_torch = seq_torch_raw[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi[
                '<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)
        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_torch_pad_all = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        seq_raw_list = self.seq_raw_torch_list[index].tolist()

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:

            # clicked_items = set(list(seq_torch_pad_all.numpy()))
            # last_clicked_item = seq_torch_pad[(1 - key_mask.int()).sum(dim=-1)-1].item()
            # target_id_neg = []
            # for _ in range(self.num_neg):
            #     target_id_neg_i = torch.randint(low=2, high=self.vocab_size, size=[]).item()
            #     while target_id_neg_i in clicked_items:
            #         target_id_neg_i = torch.randint(low=2, high=self.vocab_size, size=[]).item()
            #     target_id_neg.append(target_id_neg_i)

            all_items = list(range(2, self.vocab_size + 2))

            # all_items.remove(last_clicked_item)

            neg_items = [x for x in all_items if x not in seq_raw_list]

            target_id_neg_torch = torch.tensor(neg_items, dtype=torch.long)

            # target_id_neg_torch = torch.arange(2, self.vocab_size)

        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
        #
        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


class DatasetWithMaskProbNegSample(Dataset):
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])
        self.neg_sample_torch_arr = torch.tensor([vocab_i[token] for token in list(dict(vocab_i.freqs).keys())], dtype=torch.long)
        self.neg_sample_p_torch = torch.tensor(list(dict(vocab_i.freqs).values()), dtype=torch.float)/torch.tensor(list(dict(vocab_i.freqs).values()), dtype=torch.float).sum()

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []

        for i in range(self.seq_num):
            seq_torch = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            seq_torch = seq_torch[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi[
                '<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)

        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_all_torch_pad = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:
            target_id_neg_torch_list = []
            for h in range(seq_torch_pad.shape[0]):

                cur_neg_sample_p_torch = self.neg_sample_p_torch.clone()

                for i in range(seq_all_torch_pad.shape[0]):
                    item_torch = seq_all_torch_pad[i]
                    if item_torch != 1:  # 1 is padding
                        zero_index = (self.neg_sample_torch_arr == item_torch).nonzero(as_tuple=True)[0]
                        cur_neg_sample_p_torch[zero_index] = 0

                idx = torch.multinomial(cur_neg_sample_p_torch, self.num_neg, replacement=False)

                target_id_neg_torch_h = self.neg_sample_torch_arr.gather(0, idx)

                target_id_neg_torch_list.append(target_id_neg_torch_h)
            target_id_neg_torch = torch.stack(target_id_neg_torch_list, dim=0)

        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))

        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


class DatasetWithMaskProbNegSampleEval(Dataset):
    def __init__(self, data, vocab_u, vocab_i, num_heads=2, max_len=30, num_neg=1):

        #         self.data = data

        self.vocab_u = vocab_u
        self.vocab_i = vocab_i
        self.num_heads = num_heads
        self.max_len = max_len
        self.seq_num = len(data)
        self.num_neg = num_neg
        self.vocab_size = len([*dict(vocab_i.freqs)])
        self.neg_sample_torch_arr = torch.tensor([vocab_i[token] for token in list(dict(vocab_i.freqs).keys())], dtype=torch.long)
        self.neg_sample_p_torch = torch.tensor(list(dict(vocab_i.freqs).values()), dtype=torch.float)/torch.tensor(list(dict(vocab_i.freqs).values()), dtype=torch.float).sum()

        user_list, behavior_list = tuple(map(list, zip(*data)))

        self.user_arr = torch.tensor([vocab_u[token] for token in user_list], dtype=torch.long)

        seq_torch_pad_list = []
        pos_id_torch_list = []
        key_mask_list = []
        loss_mask_list = []
        mask_base_list = []
        for i in range(self.seq_num):
            seq_torch = torch.tensor([vocab_i[token] for token in behavior_list[i]], dtype=torch.long)
            seq_torch = seq_torch[-self.max_len:]
            seq_torch_pad = torch.ones(self.max_len, dtype=torch.long) * vocab_i.stoi[
                '<pad>']  # note that vocab.stoi['<pad>'] == 1
            seq_torch_pad[:len(behavior_list[i])] = seq_torch
            seq_torch_pad_list.append(seq_torch_pad)

            pos_id = list(range(min(len(behavior_list[i]), self.max_len)))
            pos_id.reverse()
            pos_id_torch = torch.tensor(pos_id, dtype=torch.long)
            pos_id_torch_pad = torch.zeros(self.max_len, dtype=torch.long) + 4999
            pos_id_torch_pad[:len(behavior_list[i])] = pos_id_torch
            pos_id_torch_list.append(pos_id_torch_pad)

            key_mask = (seq_torch_pad[:-1] == self.vocab_i.stoi['<pad>'])
            key_mask_list.append(key_mask)

            loss_mask = 1 - (seq_torch_pad[1:] == self.vocab_i.stoi['<pad>']).float()
            loss_mask_list.append(loss_mask)

            mask_base = (torch.triu(torch.ones(self.max_len - 1, self.max_len - 1)) == 1).transpose(0, 1).float()
            mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
            mask_base_list.append(mask_base)

        self.seq_all_torch_pad_arr = torch.stack(seq_torch_pad_list, dim=0)
        self.pos_id_all_torch_pad_arr = torch.stack(pos_id_torch_list, dim=0)

        self.seq_torch_pad_arr = self.seq_all_torch_pad_arr[:, :-1]
        self.pos_id_torch_pad_arr = self.pos_id_all_torch_pad_arr[:, :-1]

        self.key_mask_arr = torch.stack(key_mask_list, dim=0)
        self.loss_mask_arr = torch.stack(loss_mask_list, dim=0)
        self.mask_base_arr = torch.stack(mask_base_list, dim=0)

        self.target_arr = self.seq_all_torch_pad_arr[:, 1:]

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):

        #         user_id, seq, target_id = self.data[index]  #  user_id: str, seq: list(items), target_id: str

        user_id_torch = self.user_arr[index]
        seq_torch_pad = self.seq_torch_pad_arr[index]
        seq_all_torch_pad = self.seq_all_torch_pad_arr[index]
        target_id_torch = self.target_arr[index]

        target_id_neg_torch = None
        if self.num_neg > 0:

            cur_neg_sample_p_torch = self.neg_sample_p_torch.clone()

            for i in range(seq_all_torch_pad.shape[0]):
                item_torch = seq_all_torch_pad[i]
                if item_torch != 1:  # 1 is padding
                    zero_index = (self.neg_sample_torch_arr == item_torch).nonzero(as_tuple=True)[0]
                    cur_neg_sample_p_torch[zero_index] = 0

            idx = torch.multinomial(cur_neg_sample_p_torch, self.num_neg, replacement=False)

            target_id_neg_torch_h = self.neg_sample_torch_arr.gather(0, idx)
            target_id_neg_torch = target_id_neg_torch_h  # for evaluate, only sample at the last one

        # key_mask = (seq_torch_pad == self.vocab_i.stoi['<pad>'])
        #
        # loss_mask = 1 - (target_id_torch == self.vocab_i.stoi['<pad>']).float()
        #
        # mask_base = (torch.triu(torch.ones(self.max_len-1, self.max_len-1)) == 1).transpose(0, 1).float()
        #
        # mask_base = mask_base.masked_fill(mask_base <= 0, float('-inf')).masked_fill(mask_base >= 1, float(0.0))
        #
        # mask = mask_base.unsqueeze(0).repeat([self.num_heads, 1, 1])

        key_mask = self.key_mask_arr[index]
        loss_mask = self.loss_mask_arr[index]
        mask_base = self.mask_base_arr[index]

        pos_id_torch_pad = self.pos_id_torch_pad_arr[index]

        return user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask


def parse_dataset_text_line(line_str):
    #     print(line_str[1:-2])
    a_str = line_str[1:-2]
    a = line_str[1:-2].split(', ')
    #     print(a)
    user = int(a[0])
    target = int(a[-1])
    behavior_list = a_str[len(a[0]) + 2: -(len(a[-1]) + 2)]
    #     print("b:")
    #     print(behavior_list)
    #     print("b")
    res = (user, list(map(lambda x: int(x), behavior_list[1:-1].split(', '))), target)
    #     print(res)
    #     print(type(res[0]))
    #     print(type(res[1]))
    #     print(type(res[1][0]))
    #     print(type(res[2]))
    return res


def read_dataset_txt(path):
    data_set_load = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            res = parse_dataset_text_line(line)
            data_set_load.append(res)
            line = f.readline()

    return data_set_load


def load_dataset_movielens(dataset_path):
    with open(dataset_path + 'vocab_u_i.pkl', 'rb') as f:
        vocab_u = pickle.load(f)
        vocab_i = pickle.load(f)

    train_data_path = dataset_path + 'train_set.txt'
    train_set = read_dataset_txt(train_data_path)

    valid_data_path = dataset_path + 'valid_set.txt'
    valid_set = read_dataset_txt(valid_data_path)

    test_data_path = dataset_path + 'test_set.txt'
    test_set = read_dataset_txt(test_data_path)

    return train_set, valid_set, test_set, vocab_u, vocab_i


def load_dataset_amazon(dataset_path):
    with open(dataset_path, 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)
        vocab_u = pickle.load(f)
        vocab_i = pickle.load(f)

    return train_set, valid_set, test_set, vocab_u, vocab_i
