#coing:utf-8

import os
import math
#import system
import json
import random
random.seed(1234)
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datetime import datetime


def pause():
    programPause = input("Press the <ENTER> key to continue...")

def sign(x):
    if x > 0:
        return 1.
    elif x <= 0:
        return 0.
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class LookasticData(Dataset):
    def __init__(self, imgId_idx_map, train_ids, train_xgb_preds, train_pair_predIdx_map, all_paths, all_masks, treeLeafId_pathIdx_map, all_Leafnode_weight_mask, item_featureCode_map, item_featureMask_map, train_outfits=None):
        self.train_ids = train_ids
        self.train_xgb_preds = train_xgb_preds
        self.train_pair_predIdx_map = train_pair_predIdx_map
        self.imgId_idx_map = imgId_idx_map

        self.train_outfits = train_outfits
        self.all_paths = all_paths
        self.all_masks = all_masks
        self.treeLeafId_pathIdx_map = treeLeafId_pathIdx_map
        self.all_Leafnode_weight_mask  = all_Leafnode_weight_mask

        self.item_featureCode_map = item_featureCode_map
        self.item_featureMask_map = item_featureMask_map


    def generate_path_mask(self, pair_pred):
        paths = []
        masks = []
        leafnodeMasks = []
        for tree_id, leaf_id in enumerate(pair_pred):
            treeleaf_id = str(tree_id) + "_" + str(leaf_id)
            steps = self.all_paths[self.treeLeafId_pathIdx_map[treeleaf_id]]
            mask = self.all_masks[self.treeLeafId_pathIdx_map[treeleaf_id]]
            leafnodeMask = self.all_Leafnode_weight_mask[self.treeLeafId_pathIdx_map[treeleaf_id]]
            paths.append(steps)
            masks.append(mask)
            leafnodeMasks.append(leafnodeMask)

        paths = torch.stack(paths)
        masks = torch.stack(masks)
        leafnodeMasks = torch.stack(leafnodeMasks)

        return paths, masks, leafnodeMasks


    def generate_outfit_ids(self, outfit, max_len):
        outfit_ids = []
        len_ = len(outfit)

        new_outfit = None
        if len_ >= max_len:
            new_outfit = random.sample(outfit, max_len)
        else:
            new_outfit = outfit + ["pad_pad"] * (max_len - len_)
        for pair in new_outfit:
            qry_id, res_id = pair.split("_")
            qry_id = self.imgId_idx_map[qry_id]
            res_id = self.imgId_idx_map[res_id]
            outfit_ids.append([qry_id, res_id])

        return torch.LongTensor(outfit_ids)


    def __len__(self):
        return len(self.train_ids)


    def get_meta_mask(self, item_id):
        return self.item_featureCode_map[item_id], self.item_featureMask_map[item_id]


    def __getitem__(self, idx):
        qry_id, pos_id, neg_id = self.train_ids[idx]

        qry_meta, qry_mask = self.get_meta_mask(qry_id)
        pos_meta, pos_mask = self.get_meta_mask(pos_id)
        neg_meta, neg_mask = self.get_meta_mask(neg_id)

        pos_pair = qry_id + "_" + pos_id
        neg_pair = qry_id + "_" + neg_id

        # generate the path and mask of postive pair
        pos_pair_pred = self.train_xgb_preds[self.train_pair_predIdx_map[pos_pair]]
        pos_path, pos_mask, pos_leafnodeMask = self.generate_path_mask(pos_pair_pred)

        # generate the path and mask of negtive pair
        neg_pair_pred = self.train_xgb_preds[self.train_pair_predIdx_map[neg_pair]]
        neg_path, neg_mask, neg_leafnodeMask = self.generate_path_mask(neg_pair_pred)

        # generate the image features of query, postive, and negtive
        qry_id = torch.LongTensor([self.imgId_idx_map[qry_id]])
        pos_id = torch.LongTensor([self.imgId_idx_map[pos_id]])
        neg_id = torch.LongTensor([self.imgId_idx_map[neg_id]])

        # generate the imgage features of positive and negtive outfits, 
        # dimension: [N, L, 2, 2048], N is the batch size, L is the number of pairs in each outfit
        if self.train_outfits is not None:
            pos_outfit = self.train_outfits[idx % len(self.train_outfits)]["pos"]
            neg_outfit = self.train_outfits[idx % len(self.train_outfits)]["neg"]
            pos_outfit_ids = self.generate_outfit_ids(pos_outfit, 36)
            neg_outfit_ids = self.generate_outfit_ids(neg_outfit, 36)
        else:
            pos_outfit_ids = None
            neg_outfit_ids = None

        result = [qry_id, pos_id, neg_id, pos_path, pos_mask, pos_leafnodeMask, neg_path, neg_mask, neg_leafnodeMask, pos_outfit_ids, neg_outfit_ids]

        return result


class LookasticTestData(LookasticData):
    def __init__(self, imgId_idx_map, test_pairs, test_xgb_preds, test_pair_predIdx_map, all_paths, all_masks, treeLeafId_pathIdx_map, item_featureCode_map, item_featureMask_map):
        self.test_pairs = test_pairs
        self.test_xgb_preds = test_xgb_preds
        self.test_pair_predIdx_map = test_pair_predIdx_map
        self.imgId_idx_map = imgId_idx_map

        self.all_paths = all_paths
        self.all_masks = all_masks
        self.treeLeafId_pathIdx_map = treeLeafId_pathIdx_map

        self.item_featureCode_map = item_featureCode_map
        self.item_featureMask_map = item_featureMask_map


    def __len__(self):
        return len(self.test_pairs)


    def get_meta_mask(self, item_id):
        return self.item_featureCode_map[item_id], self.item_featureMask_map[item_id]


    def __getitem__(self, idx):
        pair = self.test_pairs[idx]
        qry_id, res_id = pair.split("_")
       # qry_meta, qry_mask = self.get_meta_mask(qry_id)
       # res_meta, res_mask = self.get_meta_mask(res_id)

#        qry_id, res_id = pair.split("_")

        # generate the path and mask of the pair
        pair_pred = self.test_xgb_preds[self.test_pair_predIdx_map[pair]]
        path_ids = []
        for tree_id, leaf_id in enumerate(pair_pred):
            treeleaf_id = str(tree_id) + "_" + str(leaf_id)
            path_ids.append(self.treeLeafId_pathIdx_map[treeleaf_id])
        path_ids = torch.LongTensor(path_ids)

        # path, mask = self.generate_path_mask(pair_pred)

        # generate the image features of query, postive, and negtive
        qry_id = torch.LongTensor([self.imgId_idx_map[qry_id]])
        res_id = torch.LongTensor([self.imgId_idx_map[res_id]])

        #result = [pair, qry_id, res_id, path, mask]
        result = [pair, qry_id, res_id, path_ids]

        return result


class LookasticDataset():
    def __init__(self, conf):
        data_path = conf["data_path"]
        self.data_path = data_path
        self.meta_code_map = self.get_meta_code_map() # One-hot encoding with split decision "yes", "no", 'pad'
        #self.meta_code_map = json.load(open(data_path + "/meta_code_map.json"))
        self.treeleaf_path_map = self.get_treeleaf_path_map()
        #self.treeleaf_path_map = json.load(open(data_path + "/treeleaf_path_map.json"))
        self.img_features, self.imgId_idx_map = self.get_img_features()

        self.train_ids = json.load(open(data_path + "/train_ids.json")) 
        self.test_ids = json.load(open(data_path + "/test_ids.json")) 
        self.test_dict = json.load(open(data_path + "/test_dict.json")) 
        self.test_pairs = self.generate_test_pairs(self.test_dict, conf["batch_size"])

        self.train_xgb_preds = np.load(data_path + "/train_xgb_pred_res.npy")
        self.test_xgb_preds = np.load(data_path + "/test_xgb_pred_res.npy")

        self.train_pair_predIdx_map = json.load(open(data_path + "/pair_featIdx_map_train.json"))
        self.test_pair_predIdx_map = json.load(open(data_path + "/pair_featIdx_map_test.json"))

        self.train_outfits = json.load(open(data_path + "/train_outfits.json"))

        self.batch_size = conf["batch_size"]
        self.test_batch_size = conf["test_batch_size"]
        self.meta_fea_len = len(self.meta_code_map)
        self.tree_max_depth = conf["tree_max_depth"] * 2

        self.all_paths, self.all_masks, self.treeLeafId_pathIdx_map, self.all_Leafnode_weight_mask = self.get_all_path_mask()
        print("start to get item_feature_map: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.item_featureCode_map, self.item_featureMask_map = self.get_item_feature_map(self.meta_code_map)
        print("finish to get item_feature_map: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #self.all_paths = np.load(open(data_path + "/all_paths.npy"))
        #self.all_masks = np.load(open(data_path + "/all_masks.npy"))
        #self.treeLeafId_pathIdx_map = json.load(open(data_path + "/treeLeafId_pathIdx_map.json"))

        self.train_set = LookasticData(
            self.imgId_idx_map, 
            self.train_ids, 
            self.train_xgb_preds, 
            self.train_pair_predIdx_map, 
            self.all_paths,
            self.all_masks,
            self.treeLeafId_pathIdx_map,
            self.all_Leafnode_weight_mask,
            self.item_featureCode_map,
            self.item_featureMask_map,
            train_outfits=self.train_outfits 
        )
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers= 12
        )
        self.test_set = LookasticTestData(
            self.imgId_idx_map, 
            self.test_pairs, 
            self.test_xgb_preds, 
            self.test_pair_predIdx_map, 
            self.meta_code_map, 
            self.tree_max_depth,
            self.treeLeafId_pathIdx_map,
            self.item_featureCode_map,
            self.item_featureMask_map
        )
        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            num_workers= 12
        )


    def get_all_path_mask(self):
        all_paths = []
        all_masks = []
        all_Leafnode_weight_mask = []
        treeLeafId_pathIdx_map = {}
        for treeLeafId, res in self.treeleaf_path_map.items():
            paths = res["path"]
            Leafnode_weight_mask = res["weightMask"]

            steps = []
            last_decision = None
            for path in paths:
                cur_code = path["code"]
                cur_decision = path["decision"]
                steps.append(cur_code)
                if cur_decision:
                    steps.append(self.meta_code_map["yes"])
                else:
                    steps.append(self.meta_code_map["no"]) 

            mask = [1 for i in range(len(steps))]
            if len(steps) < self.tree_max_depth:
                add_len = self.tree_max_depth - len(steps)
                steps += [self.meta_code_map["pad"] for i in range(add_len)]
                mask += [0 for i in range(add_len)]

            all_paths.append(steps)
            all_masks.append(mask)
            treeLeafId_pathIdx_map[treeLeafId] = len(all_paths) - 1
            all_Leafnode_weight_mask.append(Leafnode_weight_mask)

        all_paths = torch.LongTensor(all_paths)
        all_masks = torch.FloatTensor(all_masks)
        all_Leafnode_weight_mask = torch.FloatTensor(all_Leafnode_weight_mask)

        #np.save(self.data_path + "/all_paths.npy", all_paths)
        #np.save(self.data_path + "/all_masks.npy", all_masks)
        #json.dump(treeLeafId_pathIdx_map, open(self.data_path + "/treeLeafId_pathIdx_map.json", "w"))
        #print("Save all_paths all_masks treeLeafId_pathIdx_map successfully!")

        return all_paths, all_masks, treeLeafId_pathIdx_map, all_Leafnode_weight_mask


    def get_item_feature_map(self, meta_code_map):
        max_len = 0
        item_feature_map = json.load(open(self.data_path + "/item_feature_map.json"))
        for item_id, metas in item_feature_map.items():
            if len(metas) > max_len:
                max_len = len(metas)

        item_featureCode_map = {}
        item_featureMask_map = {}
        for item_id, metas in item_feature_map.items():
            len_ = len(metas)
            meta_code = []
            mask = [1] * len_
            if len_ < max_len:
                metas += ["pad"] * (max_len - len_)
                mask += [0] * (max_len - len_)

            meta_code = [meta_code_map[meta] for meta in metas]
            
            item_featureCode_map[item_id] = meta_code
            item_featureMask_map[item_id] = mask

        json.dump(item_featureCode_map, open(self.data_path + "/item_featureCode_map.json", "w"))
        print("Save item_featureCode_map successfully!")
        json.dump(item_featureMask_map, open(self.data_path + "/item_featureMask_map.json", "w"))
        print("Save item_featureMask_map successfully!")

        return item_featureCode_map, item_featureMask_map


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
        item_info = json.load(open(self.data_path + "/item_info_final.json"))
        item_img_map = {}
        for item_id, res in item_info.items():
            item_img_map[res["img_id"]] = item_id

        img_features = [[0]*2048]
        imgId_idx_map = {"pad": 0}
        cnt = 1
        for line in open(self.data_path + "/img_features.txt"):
            for img_id, fea in json.loads(line.strip()).items():
                item_id = item_img_map[img_id]
                imgId_idx_map[item_id] = cnt
                cnt += 1
                img_features.append(fea)

        return torch.FloatTensor(img_features), imgId_idx_map


    def get_meta_code_map(self):
        meta_list = []
        filepath = self.data_path + "/feature_map.txt"
        for line in open(filepath):
            meta_list.append(line.strip().split("\t")[1])
        meta_list += ["yes", "no", "pad"]
        meta_code_map = {}
        for i, meta in enumerate(meta_list):
            meta_code_map[meta] = i

        json.dump(meta_code_map, open(self.data_path + "/meta_code_map.json", "w"))
        print("Save meta_code_map successfully!")
        
        return meta_code_map


    def get_treeleaf_path_map(self):
        treeleaf_path_map = json.load(open(self.data_path + "/treeleaf_path_map.json"))
        for treeleaf, steps in treeleaf_path_map.items():
            treeleaf_path_map[treeleaf]["weightMask"] = steps["value"]  #sign(steps["value"])
            for i, step in enumerate(steps["path"]):
                treeleaf_path_map[treeleaf]["path"][i]["code"] = self.meta_code_map[step["str"]]
        
        json.dump(treeleaf_path_map, open(self.data_path + "/treeleaf_path_map_generated_temp.json", "w"))
        print("Save treeleaf_path_map successfully!")
#        pause()
        return treeleaf_path_map


    def cal_auc_hit_n(self, pair_scores, ns, metric):
        hit_ns = [[] for i in range(len(ns))]
        ndcg_ns = [[] for i in range(len(ns))]
        aucs = []
        for pos_pair, neg_pairs in self.test_dict.items():
            pos_score = pair_scores[pos_pair]
            neg_scores = np.array([pair_scores[neg_pair] for neg_pair in neg_pairs])

            position = (pos_score <= neg_scores).sum()

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
