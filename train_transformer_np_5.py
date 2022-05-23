"""
Transformer as neural cluster, independent transformer encoder for interests
"""


from models_language_model_style import *
from utils_language_model_style import *

import datetime
import logging
import argparse

cur_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--n_worker", type=int, default=10)
parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--dataset", type=str, default="SASRecBeauty")
parser.add_argument("--max_len", type=int, default=50)
parser.add_argument("--d_model", type=int, default=50)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--n_interests", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--r_loss", type=float, default=1.0)
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--n_neg", type=int, default=1)
parser.add_argument("--prob_neg_train", type=int, default=0)
parser.add_argument("--prob_neg_test", type=int, default=0)
parser.add_argument("--evaluate_all", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

n_worker = args.n_worker  # multiprocess environment

# dataset_path = 'dataset/Movielens1M/'
# train_set, valid_set, test_set, vocab_u, vocab_i = load_dataset_movielens(dataset_path)
# dataset_path = 'dataset/SASRecBeauty/item_only_full_LM.pkl'

dataset = args.dataset


if dataset.startswith("Movielens"):
    dataset_path = 'dataset/%s_LM/' % (dataset)
    train_set, valid_set, test_set, vocab_u, vocab_i = load_dataset_movielens(dataset_path)
else:
    dataset_path = 'dataset/%s/item_only_full_5U_LM.pkl' % (dataset)
    train_set, valid_set, test_set, vocab_u, vocab_i = load_dataset_amazon(dataset_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab_i.stoi)  # the size of vocabulary
max_seq_len = args.max_len + 1  # +1: the target item is also in the sequence


d_model = args.d_model  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = args.n_layers  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nheads = args.n_heads  # the number of heads in the multiheadattention models
ninterests = args.n_interests  # the number of interests
dropout = args.dropout  # the dropout value

model = TransformerNP5Model(ntokens, d_model, nlayers, nheads, ninterests, dropout).to(device)

lr = args.learning_rate   # learning rate
l2 = args.l2
r_loss = args.r_loss
lam = args.lam
n_neg = args.n_neg
train_prob_neg = args.prob_neg_train
test_prob_neg = args.prob_neg_test
evaluate_all = args.evaluate_all
n_epochs = args.n_epochs
batch_size = args.batch_size

log_dir = "logs_5U_EvalAll=%d/TransformerNP5_pn=%d_%s_n_interest=%s_n_layer=%s_low_temp_n_neg=%s_r_loss=%f_lam=%f_d=%d_l2=%.9f_%s_log/" % (evaluate_all, test_prob_neg, dataset, ninterests, nlayers, n_neg, r_loss, lam, d_model, l2, cur_time)
os.makedirs(log_dir, exist_ok=True)
model_training_dir = log_dir + "/model_training/"
os.makedirs(model_training_dir, exist_ok=True)

logging.basicConfig(filename=log_dir + 'training_record.log', filemode="a", level=logging.DEBUG)

# for param in model.parameters():
#     try:
#         torch.nn.init.xavier_uniform_(param.data)
#     except:
#         pass

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass # just ignore those failed init layers

from torch.utils.data import DataLoader

criterion = nn.BCEWithLogitsLoss(reduction='none')
criterion_eval = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=l2)
# item_emb_parameters = list(map(id, model.se.parameters()))
# pos_emb_parameters = list(map(id, model.pe.parameters()))
#
# base_params = filter(lambda p: id(p) not in item_emb_parameters + pos_emb_parameters,
#                      model.parameters())
# params = [{'params': base_params},
#           {'params': model.se.parameters(), 'lr': lr * 0.1},
#           {'params': model.pe.parameters(), 'lr': lr * 0.1}]
#
# optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), weight_decay=l2)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1.0)

train_batch_size = batch_size
val_batch_size = batch_size

# train_dataset = DatasetWithMask(train_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=1)

if train_prob_neg:
    train_dataset = DatasetWithMaskProbNegSample(train_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=n_neg)
else:
    train_dataset = DatasetWithMask(train_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=n_neg)

if evaluate_all:
    val_dataset = DatasetWithMaskEvalAll(valid_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=1)
    test_dataset = DatasetWithMaskEvalAll(test_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=1)
    val_batch_size = 1  # num of neg not the same for each user
else:
    if test_prob_neg:
        val_dataset = DatasetWithMaskProbNegSampleEval(valid_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=100)
        test_dataset = DatasetWithMaskProbNegSampleEval(test_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=100)
    else:
        val_dataset = DatasetWithMaskEval(valid_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=100)
        test_dataset = DatasetWithMaskEval(test_set, vocab_u, vocab_i, max_len=max_seq_len, num_neg=100)


seq_len = train_dataset.max_len

dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
dataloader_val = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_worker)
dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_worker)

import time

def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    total_aux_loss = 0.
    start_time = time.time()
    for batch, item in enumerate(dataloader_train):

        #         target_id_torch_all = torch.cat([target_id_torch.unsqueeze(1),target_id_neg_torch], axis=-1)

        optimizer.zero_grad()

        user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask = item

        seq_torch_pad = seq_torch_pad.to(device)
        mask_base = mask_base[0].to(device)
        key_mask = key_mask.to(device)

        target_id_torch = target_id_torch.to(device)
        target_id_neg_torch = target_id_neg_torch.to(device)

        pos_id_torch_pad = pos_id_torch_pad.to(device)
        loss_mask = loss_mask.to(device)

        user_emb = model(seq_torch_pad, mask_base, key_mask, pos_id_torch_pad)  # [batch_size, length, uinterest, dim]

        item_emb_p = model.se(target_id_torch).unsqueeze(2)  # [batch_size, length, 1, dim]
        item_emb_n = model.se(target_id_neg_torch).unsqueeze(2)  # [batch_size, length, 1, num_neg, dim]

        # logits_p = torch.max(torch.sum(user_emb * item_emb_p, dim=-1), dim=-1)[0]  # [batch_size, length]
        # logits_n = torch.max(torch.sum(user_emb.unsqueeze(3) * item_emb_n, dim=-1), dim=-2)[
        #     0]  # [batch_size, length, num_neg=1]

        # loss_mask = 1-key_mask.float()
        # logits_p = torch.sum(user_emb * item_emb_p, dim=-1)
        # logits_n = torch.sum(user_emb.unsqueeze(3) * item_emb_n, dim=-1)
        # loss = (criterion(logits_p, torch.ones_like(logits_p, dtype=torch.float)) * loss_mask.unsqueeze(
        #     2)).sum() / loss_mask.sum()
        # loss += (criterion(logits_n, torch.zeros_like(logits_n, dtype=torch.float)) * loss_mask.unsqueeze(-1).unsqueeze(
        #     -1)).sum() / loss_mask.sum()

        # loss = (criterion(logits_p, torch.ones_like(logits_p, dtype=torch.float))*loss_mask).sum()/loss_mask.sum()
        # loss += (criterion(logits_n, torch.zeros_like(logits_n, dtype=torch.float))*loss_mask.unsqueeze(-1)).sum()/loss_mask.sum()

        logits_p = torch.max(torch.sum(user_emb * item_emb_p, dim=-1), dim=-1)[0]  # [batch_size, length]
        logits_n = torch.max(torch.sum(user_emb.unsqueeze(3) * item_emb_n, dim=-1), dim=-2)[
            0]  # [batch_size, length, num_neg=1]

        # loss_mask_2 = 1 - key_mask.float()

        loss_p = criterion(logits_p, torch.ones_like(logits_p, dtype=torch.float))
        loss_n = criterion(logits_n, torch.zeros_like(logits_n, dtype=torch.float))

        loss_p_1 = (loss_p * loss_mask).sum()
        loss_n_1 = (loss_n * loss_mask.unsqueeze(-1)).sum()

        loss = (loss_p_1 + loss_n_1) / loss_mask.sum()

        # loss_p_2 = (loss_p * loss_mask_2).sum()
        # loss_n_2 = (loss_n * loss_mask_2.unsqueeze(-1)).sum()
        # loss_reg = (loss_p_2-loss_p_1+loss_n_2-loss_n_1)/(loss_mask_2.sum()-loss_mask.sum())
        # loss = (loss_p_1 + loss_n_1)/loss_mask.sum() + r_loss * loss_reg


        # aux_loss
        activ_matrix = model.activ_matrix[:, :, :, 0]  # [batch_size, length, ninterest]
        #         print(activ_matrix[0, :, :])
        aux_loss = -torch.mean(
            torch.sum(activ_matrix * torch.log(activ_matrix) * (1 - key_mask.unsqueeze(2).float()), dim=-1))  # entrpy

        aux_loss_2_mask = torch.matmul(1 - key_mask.unsqueeze(2).float(), 1 - key_mask.unsqueeze(1).float())

        #         print(torch.matmul(model.src_for_c, model.src_for_c.permute(0, 2, 1))[0])
        #         print(torch.matmul(activ_matrix, activ_matrix.permute(0, 2, 1))[0])

        Adj_tmp1 = model.src_for_c.unsqueeze(1).repeat(1, model.src_for_c.shape[1], 1, 1)  # [batch_size, length, length, dim]
        Adj_tmp2 = model.src_for_c.unsqueeze(2).repeat(1, 1, model.src_for_c.shape[1], 1)
        Adj = (F.cosine_similarity(Adj_tmp1, Adj_tmp2, dim=-1) + 1) / 2  # [batch_size, length, length]

        #         print(Adj.shape)
        #         print(Adj[0])

        aux_loss_2 = torch.sum(
            (Adj - torch.matmul(activ_matrix, activ_matrix.permute(0, 2, 1))) ** 2 * aux_loss_2_mask) / torch.sum(
            aux_loss_2_mask)

        # lam = 1.

        aux_loss = lam * aux_loss_2/loss_mask.sum()  # only aux_loss_2
        # aux_loss

        loss_full = loss + aux_loss
        loss_full.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        total_aux_loss += aux_loss.item()
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            cur_aux_loss = total_aux_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | auxloss {:5.4f} |'.format(
                epoch, batch, len(dataloader_train.dataset) // train_batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, cur_aux_loss))

            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | auxloss {:5.4f} |'.format(
                epoch, batch, len(dataloader_train.dataset) // train_batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, cur_aux_loss))

            total_loss = 0.
            total_aux_loss = 0.
            start_time = time.time()


def evaluate(eval_model, dataloader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    NDCG_1 = 0.0
    HIT_1 = 0.0
    NDCG_5 = 0.0
    HIT_5 = 0.0
    NDCG_10 = 0.0
    HIT_10 = 0.0
    NDCG_20 = 0.0
    HIT_20 = 0.0
    NDCG_50 = 0.0
    HIT_50 = 0.0
    MRR = 0.0
    num_samples = 0.0
    with torch.no_grad():
        for i, item in enumerate(dataloader):
            user_id_torch, seq_torch_pad, target_id_torch, target_id_neg_torch, mask_base, key_mask, pos_id_torch_pad, loss_mask = item
            seq_torch_pad = seq_torch_pad.to(device)
            mask_base = mask_base[0].to(device)
            key_mask = key_mask.to(device)

            target_id_torch = target_id_torch.to(device)
            target_id_neg_torch = target_id_neg_torch.to(device)
            pos_id_torch_pad = pos_id_torch_pad.to(device)
            loss_mask = loss_mask.to(device)
            #             print(seq_torch_pad[0])
            #             print(target_id_torch[0])
            #             print(target_id_neg_torch[0])

            user_emb = eval_model(seq_torch_pad, mask_base, key_mask,
                             pos_id_torch_pad)  # [batch_size, length, num_interest, dim]
            #             print(user_emb.shape)
            idx = (loss_mask.int()).sum(
                dim=-1) - 1  # note that idx = sum - 1, take the last two and predict the last one
            #             print(idx[0])
            user_emb = user_emb[torch.arange(0, user_emb.shape[0]), idx]  # [batch_size, num_interest, dim]

            #             print(user_emb.shape)

            item_emb_p = eval_model.se(target_id_torch[torch.arange(0, user_emb.shape[0]), idx]).unsqueeze(
                1)  # [batch_size, dim]
            #             print(item_emb_p.shape)
            item_emb_n = eval_model.se(target_id_neg_torch).unsqueeze(1)  # [batch_size, num_neg, dim]
            #             print(item_emb_n.shape)
            logits_p = torch.max(torch.sum(user_emb * item_emb_p, dim=-1), dim=-1)[0]  # [batch_size]

            logits_n = torch.max(torch.sum(user_emb.unsqueeze(2) * item_emb_n, dim=-1), dim=-2)[
                0]  # [batch_size, num_neg]

            cur_loss = criterion_eval(logits_p, torch.ones_like(logits_p, dtype=torch.float))
            cur_loss += criterion_eval(logits_n.reshape(-1), torch.zeros_like(logits_n.reshape(-1), dtype=torch.float))

            total_loss += seq_torch_pad.shape[0] * cur_loss

            predictions = torch.cat([logits_p.unsqueeze(1), logits_n], dim=-1)

            rank = torch.argsort(torch.argsort(predictions, descending=True))[:, 0]

            rank_1 = rank[rank < 1]
            NDCG_1 += torch.sum(1 / torch.log2(rank_1.float() + 2))
            HIT_1 += float(rank_1.shape[0])

            rank_5 = rank[rank < 5]
            NDCG_5 += torch.sum(1 / torch.log2(rank_5.float() + 2))
            HIT_5 += float(rank_5.shape[0])

            rank_10 = rank[rank < 10]
            NDCG_10 += torch.sum(1 / torch.log2(rank_10.float() + 2))
            HIT_10 += float(rank_10.shape[0])

            rank_20 = rank[rank < 20]
            NDCG_20 += torch.sum(1 / torch.log2(rank_20.float() + 2))
            HIT_20 += float(rank_20.shape[0])

            rank_50 = rank[rank < 50]
            NDCG_50 += torch.sum(1 / torch.log2(rank_50.float() + 2))
            HIT_50 += float(rank_50.shape[0])

            MRR += torch.sum(1.0 / (rank + 1))

            num_samples += float(user_id_torch.shape[0])

    return total_loss / len(dataloader.dataset), NDCG_1 / num_samples, HIT_1 / num_samples, NDCG_5 / num_samples, HIT_5 / num_samples, NDCG_10 / num_samples, HIT_10 / num_samples, NDCG_20 / num_samples, HIT_20 / num_samples, NDCG_50 / num_samples, HIT_50 / num_samples, MRR / num_samples


best_val_loss = float("inf")
epochs = n_epochs

# The number of epochs
best_model = None
best_hit_10 = 0.0
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    if epoch > 150 and epoch % 20 == 0:
        val_loss, val_NDCG_1, val_HIT_1, val_NDCG_5, val_HIT_5, val_NDCG_10, val_HIT_10, val_NDCG_20, val_HIT_20, val_NDCG_50, val_HIT_50, val_MRR = evaluate(model, dataloader_val)
        test_loss, test_NDCG_1, test_HIT_1, test_NDCG_5, test_HIT_5, test_NDCG_10, test_HIT_10, test_NDCG_20, test_HIT_20, test_NDCG_50, test_HIT_50, test_MRR = evaluate(model, dataloader_test)
        print('-' * 138)
        print('| epoch end {:3d} | time: {:5.2f}s | Loss_V {:5.2f} || NDCG_V@1 {:5.4f} | HIT_V@1 {:5.4f} | NDCG_V@5 {:5.4f} | HIT_V@5 {:5.4f} | NDCG_V@10 {:5.4f} | '
              'HIT_V@10 {:5.4f} | NDCG_V@20 {:5.4f} | HIT_V@20 {:5.4f} | NDCG_V@50 {:5.4f} | HIT_V@50 {:5.4f} | MRR_V@ {:5.4f} | Loss_T {:5.2f} | NDCG_T@1 {:5.4f} | HIT_T@1 {:5.4f} | NDCG_T@5 {:5.4f} | HIT_T@5 {:5.4f}  NDCG_T@10 {:5.4f} | '
              'HIT_T@10 {:5.4f} | NDCG_T@20 {:5.4f} | HIT_T@20 {:5.4f} | NDCG_T@50 {:5.4f} | HIT_T@50 {:5.4f} | MRR_T@ {:5.4f} |'.format(epoch, (time.time() - epoch_start_time), val_loss, val_NDCG_1, val_HIT_1, val_NDCG_5, val_HIT_5, val_NDCG_10, val_HIT_10, val_NDCG_20, val_HIT_20, val_NDCG_50, val_NDCG_50, val_MRR, test_loss, test_NDCG_1, test_HIT_1, test_NDCG_5, test_HIT_5, test_NDCG_10, test_HIT_10, test_NDCG_20, test_HIT_20, test_NDCG_50, test_HIT_50, test_MRR))
        print('-' * 138)

        logging.info('-' * 138)
        logging.info('| epoch end {:3d} | time: {:5.2f}s | Loss_V {:5.2f} || NDCG_V@1 {:5.4f} | HIT_V@1 {:5.4f} | NDCG_V@5 {:5.4f} | HIT_V@5 {:5.4f} | NDCG_V@10 {:5.4f} | '
              'HIT_V@10 {:5.4f} | NDCG_V@20 {:5.4f} | HIT_V@20 {:5.4f} | NDCG_V@50 {:5.4f} | HIT_V@50 {:5.4f} | MRR_V@ {:5.4f} | Loss_T {:5.2f} | NDCG_T@1 {:5.4f} | HIT_T@1 {:5.4f} | NDCG_T@5 {:5.4f} | HIT_T@5 {:5.4f}  NDCG_T@10 {:5.4f} | '
              'HIT_T@10 {:5.4f} | NDCG_T@20 {:5.4f} | HIT_T@20 {:5.4f} | NDCG_T@50 {:5.4f} | HIT_T@50 {:5.4f} | MRR_T@ {:5.4f} |'.format(epoch, (time.time() - epoch_start_time), val_loss, val_NDCG_1, val_HIT_1, val_NDCG_5, val_HIT_5, val_NDCG_10, val_HIT_10, val_NDCG_20, val_HIT_20, val_NDCG_50, val_NDCG_50, val_MRR, test_loss, test_NDCG_1, test_HIT_1, test_NDCG_5, test_HIT_5, test_NDCG_10, test_HIT_10, test_NDCG_20, test_HIT_20, test_NDCG_50, test_HIT_50, test_MRR))
        logging.info('-' * 138)

        best_val_loss = val_loss
        best_model = model

        if best_hit_10 <= val_HIT_10:
            best_hit_10 = val_HIT_10
            best_model = model
            torch.save(best_model, model_training_dir + '/best_model.pt')



        # scheduler.step(val_loss)
    scheduler.step()
# torch.save(best_model, model_training_dir + 'best_model.pt')
