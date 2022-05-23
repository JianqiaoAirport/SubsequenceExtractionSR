import numpy as np
import pandas as pd

import copy
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default="SASRecM1M")
parser.add_argument("--raw_file", type=str, default="ml-1m.txt")

args = parser.parse_args()

dataset = args.dataset
file_name = args.raw_file

dataset_path = 'dataset/%s/' % dataset
df = pd.read_csv(dataset_path +'raw/%s' % file_name, sep=' ', header=None, names=['userID', 'itemID'])

user_list = list(df['userID'])
# set of all the items, for negative sampling
item_list = list(df['itemID'])
from collections import Counter
from torchtext.vocab import Vocab

counter_u = Counter()
for item in user_list:
    counter_u.update([item])
vocab_u = Vocab(counter_u)

counter_i = Counter()
for item in item_list:
    counter_i.update([item])
vocab_i = Vocab(counter_i)

user_arr = df['userID'].to_numpy()
item_arr = df['itemID'].to_numpy()

train_set = []
valid_set = []
test_set = []

dropped_user_count = 0

user_id = user_arr[0]
user_sample_list = []
user_item_seq = []


for i in tqdm(range(user_arr.shape[0])):

    user_item_seq.append(item_arr[i])

    if len(user_item_seq) >= 1:
        item_seq = copy.deepcopy(user_item_seq)
        user_sample_list.append((user_id, item_seq[:]))

    if i == user_arr.shape[0] - 1 or user_arr[i + 1] != user_id:
        if len(user_item_seq) <= 4:

            dropped_user_count += 1
            print('user dropped: %d' % dropped_user_count)
        else:

            if len(user_sample_list) < 3:
                train_set.append(user_sample_list[-1])
            else:
                test_set.append(user_sample_list[-1])
                valid_set.append(user_sample_list[-2])
                train_set.append(user_sample_list[-3])

        user_item_seq = []
        user_time_seq = []
        user_sample_list = []

        try:
            user_id = user_arr[i + 1]
        except IndexError:
            print('Finished.')


import pickle
import os

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

with open(dataset_path+'item_only_full_5U_LM.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(vocab_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(vocab_i, f, pickle.HIGHEST_PROTOCOL)
