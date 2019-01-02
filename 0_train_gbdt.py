import re
import os
import json
import numpy as np
import xgboost as xgb

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def train(data_path, conf, data_path_out):
    data_name = data_path.split("/")[2]
    data_path_father = data_path.split("/")[0] + "/" + data_path.split("/")[1] + "/" + data_path.split("/")[2]

    dtrain = xgb.DMatrix(data_path + "/xgb_train.txt")
    #dtest = xgb.DMatrix(data_path + "/xgb_test.txt")
    dval = xgb.DMatrix(data_path + "/xgb_val.txt")    
    param = {
        'max_depth': conf["tree_max_depth"], 
        'eta': 1, 
        'silent': 1, 
        'objective': 'binary:logistic',
        'nthread': 10,
        'eval_metric': 'logloss'
    }
    
    watchlist = [(dval, 'eval'), (dtrain, 'train')]
    num_round = conf["xgb_num_round"]
    bst = xgb.train(param, dtrain, num_round, watchlist, verbose_eval=20)
    
    #preds_test = bst.predict(dtest)
    preds = bst.predict(dval)

    labels = dval.get_label()
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

    bst.save_model(data_path_out + "/xgbst.model")
    bst.dump_model(data_path_out + "/xgbst.dump.raw.txt")
    bst.dump_model(data_path_out + "/xgbst.dump.nice.txt", data_path_father + "/feature_map.txt")
    print("finish save and dump model")


def get_leaf_path(lines):
    all_paths = {}
    stack = []
    for node in lines:
        node_id = node.split(":")[0]
        if "leaf=" in node:
            value = float(node.split("=")[-1])
            if stack[-1]["childs"] == 2:
                stack[-1]["decision"] = False
            if stack[-1]["childs"] == 1:
                stack[-1]["decision"] = True
            

            all_paths[node_id] = {"path": [{"id": i["id"], "str": i["str"], "decision": i["decision"]} for i in stack], "value": value}

            stack[-1]["childs"] -= 1

            while stack[-1]["childs"] == 0:
                del stack[-1]
                if len(stack) != 0:
                    stack[-1]["childs"] -= 1
                    if stack[-1]["childs"] == 1:
                        stack[-1]["decision"] = True

                else:
                    break
        else:
            node_str = re.split(r"[\[\]]", node)[1]
            stack.append({"id": node_id, "str": node_str, "decision": False, "childs": 2})
    return all_paths


def parse_trees(tree_filepath):
    lines = []
    for line in open(tree_filepath):
        lines.append(line.strip())

    all_leaf_path = {}

    tree_id = 0
    tmp_tree_lines = []
    for line in lines[1:]:
        if "booster" in line:
            leaf_path_map = get_leaf_path(tmp_tree_lines)
            for leaf_id, path in leaf_path_map.items():
                all_leaf_path[str(tree_id) + "_" + leaf_id] = path

            tmp_tree_lines = []
            tree_id = re.split(r"[\[\]]", line)[1]
        else:
            tmp_tree_lines.append(line)

    # process the last tree
    leaf_path_map = get_leaf_path(tmp_tree_lines)
    for leaf_id, path in leaf_path_map.items():
        all_leaf_path[str(tree_id) + "_" + leaf_id] = path

    return all_leaf_path


def generate_pair_path(bst, data, xgb_pred_filepath, xgb_pred_score_filepath):
    #paths = []
    preds = bst.predict(data, pred_leaf=True)
    print("start to save xgb pred to file, lines: %d" %(len(preds)))
    np.save(xgb_pred_filepath, preds)
    print("save finish 1")
    
    preds_2 = bst.predict(data, pred_leaf=False)
    print("start to save xgb pred score to file, lines: %d" %(len(preds_2)))
    np.save(xgb_pred_score_filepath, preds_2)
    print("save finish 2 (score)")

def pred_leaf_idx(data_path, data_path_out):
    data_name = data_path.split("/")[2]
    
    model_path = data_path_out + "/xgbst.model"
    model_dump_path = data_path_out + "/xgbst.dump.nice.txt"

    dtrain = xgb.DMatrix(data_path + "/xgb_train.txt")
    dtest = xgb.DMatrix(data_path + "/xgb_test.txt")
    dval = xgb.DMatrix(data_path + "/xgb_val.txt")
    
    bst = xgb.Booster(model_file=model_path)
    treeleaf_path_map = parse_trees(model_dump_path)
    json.dump(treeleaf_path_map, open(data_path_out + "/treeleaf_path_map.json", "w"))

    generate_pair_path(bst, dtrain, data_path_out + "/train_xgb_pred_res", data_path_out + "/train_xgb_pred_score")
    generate_pair_path(bst, dtest, data_path_out + "/test_xgb_pred_res", data_path_out + "/test_xgb_pred_score")
    generate_pair_path(bst, dval, data_path_out + "/val_xgb_pred_res", data_path_out + "/val_xgb_pred_score")

def mkdir(path):
 
	folder = os.path.exists(path)
	if not folder:                
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")

	else:
		print("---  There is this folder!  ---")


def main():
    conf = json.load(open("./config.json"))
    datapath = "./data/men/allEva" 
    dataname = datapath.split("/")[2]
    tree_max_depth = conf["men"]["tree_max_depth"]
    xgb_num_round = conf["men"]["xgb_num_round"]

    print("dataset is ", dataname, " evaluation mode is ", datapath.split("/")[-1], "tree_max_depth: ", tree_max_depth, "xgb_num_round: ", xgb_num_round)
    pause()
    datapath_out = datapath + "/"+ str(xgb_num_round)+"_"+str(tree_max_depth) 
#    print(datapath_out)
#    pause()
    mkdir(datapath_out)
    train(datapath, conf["men"], datapath_out)
    pred_leaf_idx(datapath, datapath_out)

    datapath = "./data/women/allEva"
    dataname = datapath.split("/")[2]
    
    tree_max_depth = conf["women"]["tree_max_depth"]
    xgb_num_round = conf["women"]["xgb_num_round"]
    print("dataset is ", dataname, " evaluation mode is ", datapath.split("/")[-1], "tree_max_depth: ", tree_max_depth, "xgb_num_round: ", xgb_num_round)
    #pause()
    datapath_out = datapath + "/"+ str(xgb_num_round)+"_"+str(tree_max_depth)
    mkdir(datapath_out)

    train(datapath, conf["women"], datapath_out)
    pred_leaf_idx(datapath, datapath_out)


if __name__ == "__main__":
    main()
