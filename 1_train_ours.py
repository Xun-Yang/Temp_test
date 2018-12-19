from __future__ import division

import os
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler

from datetime import datetime

from models import *
from utility import LookasticDataset 

import sys
import json
import numpy as np
import pickle as pkl

from tensorboard_logger import configure, log_value


CUDA = True if torch.cuda.is_available() else False

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default="women", help="which dataset to use, option in: men, women")
    parser.add_argument("-m", "--mode", default="image_only", help="options in: image_only, meta_only, fusion")
    args = parser.parse_args()
    return args


def train(dataset_name, conf):
    conf = conf[dataset_name]
    print("Start Dataset Loading", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset = LookasticDataset(conf)
    print("Dataset Loading finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#    pause()

    model = ExpMatch(dataset.img_features, len(dataset.meta_code_map), conf["num_hidden"], conf["metric"], conf["loss"], conf["tree_pooling"],conf["tree_attention"], conf["path_encoding"], conf["regularization_para"], dataset.all_paths, dataset.all_masks, dataset.all_Leafnode_weight_mask)
    model.cuda()

    lr = conf["lr"]
    optimizer = torch.optim.SGD([
        {'params': model.imageW.parameters(), 'lr': lr},
        {'params': model.W_att.parameters(), 'lr': lr},
        {'params': model.h_att.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': model.predict.parameters(), 'lr': lr},
        {'params': model.meta_embed.parameters(), 'lr': lr},
    ], lr=lr, momentum=conf["momentum"], weight_decay = conf["weight_decay"])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf["lr_decay_interval"]*len(dataset.train_set)/conf["batch_size"]), gamma=conf["lr_decay_gamma"])

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    save_point = os.path.join("./checkpoint", dataset_name)
    if not os.path.isdir(save_point):
        os.mkdir(save_point)

    # initialize and configure tensorboard
    configure(os.path.join(save_point, datetime.now().strftime("%Y-%m-%d_%H_%M_%S")), flush_secs=5)

    best_auc = 0.0
    best_hit10 = 0.0
    best_score = 0.0
    best_hit20 = 0.0
    best_ndcg20 = 0.0
    
    test_interval_steps = int(float(conf["test_interval"]) * len(dataset.train_loader))
    for epoch in range(conf["epochs"]):
        for batch_cnt, batch in enumerate(dataset.train_loader): 
            dataset.train_or_test = "train"
            step = int(batch_cnt + epoch*len(dataset.train_loader) + 1)

            model.cuda()
            model.train(True)
            exp_lr_scheduler.step()
            optimizer.zero_grad()

            qry_id, pos_id, neg_id, pos_path, pos_mask, pos_leafnodeMask, neg_path, neg_mask, neg_leafnodeMask, pos_outfit_ids, neg_outfit_ids = batch
            qry_id = Variable(qry_id).cuda()
            pos_id = Variable(pos_id).cuda()
            neg_id = Variable(neg_id).cuda()
            pos_path = Variable(pos_path).cuda()
            pos_mask = Variable(pos_mask).cuda()
            neg_path = Variable(neg_path).cuda()
            neg_mask = Variable(neg_mask).cuda()
            pos_leafnodeMask = Variable(pos_leafnodeMask).cuda()
            neg_leafnodeMask = Variable(neg_leafnodeMask).cuda()

            batchsize = qry_id.shape[0]
#            print(batchsize)
            if conf["outfit_usage"]:
                pos_outfit_ids = Variable(pos_outfit_ids).cuda()
                neg_outfit_ids = Variable(neg_outfit_ids).cuda()

                pair_loss, outfit_loss = model(qry_id, pos_id, neg_id, pos_path, pos_mask, pos_leafnodeMask, neg_path, neg_mask, neg_leafnodeMask, pos_outfit_ids, neg_outfit_ids)

                pair_loss /= float(batchsize)
                outfit_loss /= float(batchsize)
                log_value("pair_loss", pair_loss, step)
                log_value("outfit_loss", outfit_loss, step)
            else:
                pair_loss = model(qry_id, pos_id, neg_id, pos_path, pos_mask, pos_leafnodeMask, neg_path, neg_mask, neg_leafnodeMask, None, None)
               
                pair_loss /= float(batchsize)
                log_value("pair_loss", pair_loss, step)
                log_value("outfit_loss", 0, step)
                outfit_loss = 0

            loss = pair_loss + outfit_loss

            if (batch_cnt+1) % 10 == 0:
            #    output_str = "%s: epoch: %d, batch: %d/%d, pair_loss: %f, outfit_loss: %f, loss: %f" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, len(dataset.train_loader), batch_cnt+1, pair_loss, outfit_loss, loss)
                output_str = "%s: epoch: %d, batch: %d/%d, loss: %f" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, len(dataset.train_loader), batch_cnt+1, loss)
                print(output_str)
                
            loss.backward()
            optimizer.step()

            if step % test_interval_steps == 0:
                print("%s: epoch %d, batch %d, loss %f" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, batch_cnt, loss))
                print("\nstart to test")
                model.eval()
                auc, hit_ns, ndcg_ns = test(model, dataset, conf)
                hit5, hit10, hit20, hit40 = hit_ns
                ndcg5, ndcg10, ndcg20, ndcg40 = ndcg_ns
                log_value("auc", auc, step)
                log_value("hit5", hit5, step)
                log_value("hit10", hit10, step)
                log_value("hit20", hit20, step)
                log_value("hit40", hit40, step)
                log_value("ndcg5", ndcg5, step)
                log_value("ndcg10", ndcg10, step)
                log_value("ndcg20", ndcg20, step)
                log_value("ndcg40", ndcg40, step)

                print("test finished\n%s: epoch %d" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch))
                print("test auc: %f,\n    hit5: %f, hit10: %f, hit20: %f, hit40: %f,\n    ndcg5: %f, ndcg10: %f, ndcg20: %f, ndcg40: %f" %(auc, hit5, hit10, hit20, hit40, ndcg5, ndcg10, ndcg20, ndcg40))
                if ndcg20 >= best_ndcg20 and hit20 >= best_hit20:
                    best_ndcg20 = ndcg20
                    best_hit20 = hit20
                    print("ACHIEVE BEST PERFORMANCE: ndcg20 %f, hit20 %f" %(best_ndcg20, best_hit20))
                    state = {
                        'model': model,
                        'ndcgs': ndcg_ns,
                        'hit_ns': hit_ns,
                        'auc': auc,
                        'epoch': epoch
                    }
                    torch.save(state, os.path.join(save_point, '%s.t7' %(dataset_name)))
                print("\n\n")

    print("\n\nTHE OVERALL BEST PERFORMANCE: ndcg20 %f, hit20 %f" %(best_ndcg20, best_hit20))


def test(model, dataset, conf):
    model.eval()
    dataset.train_or_test = "test"
    pair_score = {}
    all_path_embed = model.generate_all_path_embed()
    print("start extract test embed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for batch_cnt, batch in enumerate(dataset.test_loader):
        #print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(dataset.test_loader), batch_cnt)
        pair, qry_id, res_id, path_ids = batch
 #       print("Round start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        qry_id = Variable(qry_id).cuda()
        res_id = Variable(res_id).cuda()
        path_ids = Variable(path_ids).cuda()
        score = model.compute_pairwise_score(qry_id, res_id, path_ids, all_path_embed)
        score = score.data.cpu().numpy()
        for each_pair, each_score in zip(pair, score):
            pair_score[each_pair] = each_score

#        print("Round end time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("test embed extract finish", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ns = [5, 10, 20, 40]
    auc, hit_ns, ndcg_ns = dataset.cal_auc_hit_n(pair_score, ns, conf["metric"])
    print("auc, hit rate, ndcg caldulate finish", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return auc, hit_ns, ndcg_ns



def main():
    config = json.load(open("./config.json"))

    paras = get_cmd()
    dataset_name = paras.dataset_name
    #mode = paras.mode

    train(dataset_name, config)


if __name__ == "__main__":
    main() 
