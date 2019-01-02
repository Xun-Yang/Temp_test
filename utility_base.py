#coing:utf-8

import os
import math
#import system
import json
import random
random.seed(1234)
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _cal_score(h, t, metric):
    if metric == "Distance":
        return np.power(h - t, 2).sum(axis=-1)
    if metric == "InnerProduct":
        tmp = np.inner(t, h)
        return tmp 


def pause():
    programPause = input("Press the <ENTER> key to continue...")


class LookasticData(Dataset):
    def __init__(self, imgId_idx_map, train_ids, item_cat_map, rel_id_map, transE=False):
        self.train_ids = train_ids
        self.imgId_idx_map = imgId_idx_map
        self.item_cat_map = item_cat_map
        self.rel_id_map = rel_id_map
        self.transE = transE


    def __len__(self):
        return len(self.train_ids)


    def __getitem__(self, idx):
        ori_qry_id, ori_pos_id, ori_neg_id = self.train_ids[idx]

        # generate the image features of query, postive, and negtive
        qry_id = torch.LongTensor([self.imgId_idx_map[ori_qry_id]])
        pos_id = torch.LongTensor([self.imgId_idx_map[ori_pos_id]])
        neg_id = torch.LongTensor([self.imgId_idx_map[ori_neg_id]])

        if not self.transE:
            result = [qry_id, pos_id, neg_id]
        else:
            qry_cat = self.item_cat_map[ori_qry_id]
            pos_cat = self.item_cat_map[ori_pos_id]
            neg_cat = self.item_cat_map[ori_neg_id]

            pos_rel = "::".join([qry_cat, pos_cat])
            neg_rel = "::".join([qry_cat, neg_cat])
            if pos_rel not in self.rel_id_map:
                pos_rel = "others"
            if neg_rel not in self.rel_id_map:
                neg_rel = "others"

            pos_rel_id = self.rel_id_map[pos_rel]
            pos_rel_id = torch.LongTensor([pos_rel_id])

            neg_rel_id = self.rel_id_map[neg_rel]
            neg_rel_id = torch.LongTensor([neg_rel_id])

            result = [qry_id, pos_id, neg_id, pos_rel_id, neg_rel_id]

        return result



class LookasticTestData(LookasticData):
    def __init__(self, imgId_idx_map, test_pairs):
        self.test_pairs = test_pairs
        self.imgId_idx_map = imgId_idx_map


    def __len__(self):
        return len(self.test_pairs)


    def __getitem__(self, idx):
        pair = self.test_pairs[idx]
        ori_qry_id, ori_res_id = pair.split("_")

        # generate the image features of query, postive, and negtive
        qry_id = torch.LongTensor([self.imgId_idx_map[ori_qry_id]])
        res_id = torch.LongTensor([self.imgId_idx_map[ori_res_id]])

        result = [pair, qry_id, res_id]

        return result


class LookasticDataset():
    def __init__(self, conf):
        data_path = conf["data_path"]
        self.data_path = data_path
        self.data_path_father = data_path.split("/")[0] + "/" + data_path.split("/")[1] + "/" + data_path.split("/")[2]
        self.img_features, self.imgId_idx_map = self.get_img_features()

        self.train_ids = json.load(open(data_path + "/train_ids.json")) 
        self.test_ids = json.load(open(data_path + "/test_ids.json")) 
        self.test_dict = json.load(open(data_path + "/test_dict.json")) 
        self.test_pairs = self.generate_test_pairs(self.test_dict, conf["batch_size"])

        self.train_pair_predIdx_map = json.load(open(data_path + "/pair_featIdx_map_train.json"))
        self.test_pair_predIdx_map = json.load(open(data_path + "/pair_featIdx_map_test.json"))

        self.batch_size = conf["batch_size"]
        self.test_batch_size = conf["test_batch_size"]

        self.item_cat_map = json.load(open(self.data_path_father + "/transE_item_cat_map.json"))
        self.rel_id_map = self.get_rel_id_map(self.data_path_father)
        self.model = conf["model"]
        self.transE = True if conf["model"] == "transE" else False
        self.rel_num = len(self.rel_id_map)

        self.train_set = LookasticData(
            self.imgId_idx_map, 
            self.train_ids,
            self.item_cat_map,
            self.rel_id_map,
            self.transE
        )
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers= 6
        )
        if conf["model"] in {"transE"}:
            self.test_dict = self.get_test_dict(self.test_ids)
            self.test_qry_list, self.test_idx_list, self.test_rel_idx = self.convert_dict2list(self.imgId_idx_map, self.test_dict)
        else:
            self.test_set = LookasticTestData(
                self.imgId_idx_map, 
                self.test_pairs, 
               # self.item_cat_map,
            )
            self.test_loader = DataLoader(
                self.test_set, 
                batch_size=self.test_batch_size, 
                shuffle=False, 
                num_workers=12
            )


    def convert_dict2list(self, strId_idx_map, test_dict):
        test_qry_list = []
        test_idx_list = []
        rel_idx_list = []
        for pos_pair, negs in test_dict.items():
            tmp_list = []
            tmp_rel_list = []
            q, p = pos_pair.split("_")
            test_qry_list.append(strId_idx_map[q])
            tmp_list.append(strId_idx_map[p])

            q_cat = self.item_cat_map[q]
            p_cat = self.item_cat_map[p]
            pos_rel = "::".join([q_cat, p_cat])
            if pos_rel not in self.rel_id_map:
                pos_rel = "others"
            tmp_rel_list.append(self.rel_id_map[pos_rel])
            for neg in negs:
                neg_cat = self.item_cat_map[neg]
                neg_rel = "::".join([q_cat, neg_cat])
                if neg_rel not in self.rel_id_map:
                    neg_rel = "others"
                tmp_rel_list.append(self.rel_id_map[neg_rel])
                tmp_list.append(strId_idx_map[neg])
            test_idx_list.append(tmp_list)
            rel_idx_list.append(tmp_rel_list)

        return np.array(test_qry_list), np.array(test_idx_list), np.array(rel_idx_list)


    def get_test_dict(self, test_ids):
        test_dict = {}
        for q, p, n in test_ids:
            pos_pair = q + "_" + p
            if pos_pair not in test_dict:
                test_dict[pos_pair] = {n}
            else:
                test_dict[pos_pair].add(n)

        return test_dict


    def get_rel_id_map(self, data_path):
        rel_id_map = {}
        id_ = 0
        for line in open(data_path + "/transE_all_rel.txt"):
            rel = line.strip()
            rel_id_map[rel] = id_
            id_ += 1

        return rel_id_map


    def generate_test_pairs(self, test_dict, batch_size):
        all_pairs = set()
        for pos_pair, neg_pairs in test_dict.items():
            all_pairs.add(pos_pair)
            for neg_pair in neg_pairs:
                all_pairs.add(neg_pair)

        all_pairs = list(all_pairs)
        if len(all_pairs) % batch_size != 0:
            all_pairs += all_pairs[0: batch_size - len(all_pairs) % batch_size]

        return all_pairs


    def get_img_features(self):
        item_info = json.load(open(self.data_path_father + "/item_info_final.json"))
        item_img_map = {}
        for item_id, res in item_info.items():
            item_img_map[res["img_id"]] = item_id

        img_features = [[0]*2048]
        imgId_idx_map = {"pad": 0}
        cnt = 1
        for line in open(self.data_path_father + "/img_features.txt"):
            for img_id, fea in json.loads(line.strip()).items():
                item_id = item_img_map[img_id]
                imgId_idx_map[item_id] = cnt
                cnt += 1
                img_features.append(fea)

        return torch.FloatTensor(img_features), imgId_idx_map


    def cal_auc_hit_n(self, pair_scores, ns, metric):
        hit_ns = [[] for i in range(len(ns))]
        ndcg_ns = [[] for i in range(len(ns))]
        aucs = []
        for pos_pair, neg_pairs in self.test_dict.items():
            pos_score = pair_scores[pos_pair]
            neg_scores = np.array([pair_scores[neg_pair] for neg_pair in neg_pairs])

            if metric == "InnerProduct":
                position = (neg_scores >= pos_score).sum()
            if metric == "Distance":
                position = (neg_scores <= pos_score).sum()

            for i, _K in enumerate(ns):
                hr = position < _K
                hit_ns[i].append(hr)
                if hr:
                    ndcg_ns[i].append(math.log(2) / math.log(position+2))
                else:
                    ndcg_ns[i].append(0)
            auc = 1 - (position * 1. / len(neg_scores))
            aucs.append(auc)

        hitn = [np.array(x).mean(axis=0) for x in hit_ns]
        ndcg = [np.array(x).mean(axis=0) for x in ndcg_ns]
        auc = np.array(aucs).mean(axis=0)

        return auc, hitn, ndcg


    def cal_auc_hit_n_transE(self, rel_embeds, nodes_embed, ns, metric):
        hit_ns = [[] for i in range(len(ns))]
        ndcg_ns = [[] for i in range(len(ns))]
        aucs = []
        for qry, res, rel in zip(self.test_qry_list, self.test_idx_list, self.test_rel_idx):
            query_embed = nodes_embed[qry]
            pos_negs_embeds = nodes_embed[res]
            pos_negs_rel_embeds = rel_embeds[rel]

            if self.model == "transE":
                scores = _cal_score(query_embed + pos_negs_rel_embeds, pos_negs_embeds, metric)
            if self.model == "BPR":
                scores = _cal_score(query_embed, pos_negs_embeds, metric)
            pos_score, neg_scores = scores[0], scores[1:]

            if metric == "Distance":
                position = (neg_scores <= pos_score).sum()
            if metric == "InnerProduct":
                position = (neg_scores >= pos_score).sum()

            for i, _K in enumerate(ns):
                hr = position < _K
                hit_ns[i].append(hr)
                if hr:
                    ndcg_ns[i].append(math.log(2) / math.log(position+2))
                else:
                    ndcg_ns[i].append(0)
            auc = 1 - (position * 1. / len(neg_scores))
            aucs.append(auc)

        hitn = [np.array(x).mean(axis=0) for x in hit_ns]
        ndcg = [np.array(x).mean(axis=0) for x in ndcg_ns]
        auc = np.array(aucs).mean(axis=0)

        return auc, hitn, ndcg
