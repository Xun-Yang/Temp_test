# We generate the data for each dataset (men and women)
# The files of each dataset are saved at an individual folders in the current directory: ./men, ./women
# For each dataset, we generate one file:
#  1. item_info.json: the meta_info of each item, include tags(category, attributes from both lookastic and visenze api), image_id(to locate the corresponding image file, note that different items may correspond to the same image, so we don't use the image_id as the id)

import os
import re
import json
import random
random.seed(1)
import hashlib
import argparse



def pause():
    programPause = input("Press the <ENTER> key to continue...")


def get_cat(item_info, item_id):
    firstcat = item_info[item_id]["new_cat"][1]
    subcat = item_info[item_id]["new_cat"][2]
    if subcat in ["Blouses", "Shirts", "T-shirts", "T_shirts", "Sweaters", "Bikini_Top","Bikini_Tops"]:
        cat = "Top"
    elif subcat in ["Blazers", "Outerwear", "Coats", "Jackets"]:
        cat = "Outerwear"
    elif subcat in ["Skirts", "Pants", "Shorts", "Bikini_Pants"]:
        cat = "Bottom"
    elif subcat in ["Suits", "Track_Suits", "Dresses", "Jumpsuits", "Beach_Dresses", "Swimsuits", "Cover_ups"]:
        cat = "Fullbody"
    elif firstcat == "Footwear":
        cat = "Footwear"
    elif firstcat == "Accessories" or firstcat == "Headwear":
        cat = "Accessories"
    else:
        print("Firstcat: ", firstcat)
        print("Subcat: ", subcat)
        pause()
    return cat


def get_transE_cat(item_info, item_id):
    cat = item_info[item_id]["new_cat"][1]
    if cat in {"Top", "Bottom", "Footwear"}:
        cat = item_info[item_id]["new_cat"][2]
    if cat == "Accessories":
        sec_cat = item_info[item_id]["new_cat"][2]
        if sec_cat == "Bags":
            cat = sec_cat
    return cat


def get_cat_2(item_info, item_id):
    cat = item_info[item_id]["new_cat"][2]
    return cat


def get_img_attr_set(data_path):
    attr_set = set()
    for line in open(data_path + "/attr_list.txt"):
        attr = line.strip().split(" ")[-1]
        attr_set.add(attr)

    return attr_set


def get_img_style_map():
    img_style_tag_map = {}
    for line in open("./items_tagging_res.txt"):
        data = json.loads(line.strip())
        for img_id, res in data.items():
            for each_res in res["result"]:
                if each_res["tag_group"] == "fashion_style":
                    for each_obj in each_res["objects"]:
                        each_obj_style = []
                        for style_tag in each_obj["tags"]:
                            style_tag = style_tag["tag"]
                            each_obj_style.append(style_tag)
                        if img_id not in img_style_tag_map:
                            img_style_tag_map[img_id] = [each_obj_style]
                        else:
                            img_style_tag_map[img_id].append(each_obj_style)

    return img_style_tag_map


def get_tag_type_map(data_path):
    type_tags_map = json.load(open(data_path + "/tag_type.txt"))
    
    tag_type_map = {}
    for typee, tags in type_tags_map.items():
        for tag in tags:
            tag_type_map[tag] = typee

    return tag_type_map


def get_img_attr_map(data_path):
    attr_set = get_img_attr_set(data_path)

    img_attr_map = {}
    for line in open("./items_tagging_res.txt"):
        data = json.loads(line.strip())
        for img_id, res in data.items():
            for each_res in res["result"]:
                if each_res["tag_group"] == "fashion_attributes":
                    for each_obj in each_res["objects"]:
                        each_obj_attr = {}
                        for i in each_obj["tags"]:
                            tag_key, tag_value = i["tag"].split(":")
                            if tag_key not in attr_set:
                                continue
                            else:
                                each_obj_attr[tag_key] = tag_value
                        if img_id not in img_attr_map:
                            img_attr_map[img_id] = [each_obj_attr] 
                        else:
                            img_attr_map[img_id].append(each_obj_attr)

    return img_attr_map


def get_img_visenze_cat(data_path):
    attr_set = get_img_attr_set(data_path)

    img_visenze_cat_map = {}
    for line in open("./items_tagging_res.txt"):
        data = json.loads(line.strip())
        for img_id, res in data.items():
            for each_res in res["result"]:
                if each_res["tag_group"] == "fashion_attributes":
                    for each_obj in each_res["objects"]:
                        visenze_tax = []
                        for i in each_obj["tags"]:
                            tag_key, tag_value = i["tag"].split(":")
                            if tag_key not in attr_set:
                                visenze_tax.append(i["tag"])
                        visenze_tax = "__".join(visenze_tax)
                        if img_id not in img_visenze_cat_map:
                            img_visenze_cat_map[img_id] = [visenze_tax]
                        else:
                            img_visenze_cat_map[img_id].append(visenze_tax)

    return img_visenze_cat_map 


def get_visenze_cat_map(data_path):
    visenze_looka_cat_map = {}
    for line in open(data_path + "/visenze_cat_map_xun.txt"):
        data = line.strip().split(" ")
        vi_cat = data[-1]
        lo_cat = " ".join(data[:-1])
        visenze_looka_cat_map[vi_cat] = lo_cat

    return visenze_looka_cat_map


def preprocess_metadata(data_path):
    tag_type_map = get_tag_type_map(data_path)
    img_style_map = get_img_style_map()
    img_attr_map = get_img_attr_map(data_path)
    img_visenze_cat_map = get_img_visenze_cat(data_path)
    vi_lo_cat_map = get_visenze_cat_map(data_path)

    item_info = json.load(open(data_path + "/item_info_new_cat.json"))
    outfit_items = {}

    ttl_cnt = 0
    bad_num_cnt = 0
    correct_cat_stat = {}
    for item_id, res in item_info.items():
        keep_flag = True
        img_id = res["img_id"]
#        print("item_id: ", item_id)
        pre_cat = get_cat(item_info, item_id)#Coarse category
#        print("pre_cat: ", pre_cat)
        pre_cat_2 = get_cat_2(item_info, item_id)# The third category
        if img_id not in img_style_map:
            styles = []
        else:
            styles = img_style_map[img_id]

#        print("img_style_map[img_id]: ", img_style_map[img_id])
        attrs = img_attr_map[img_id]
#        print("attrs: ", attrs)
        vi_cats = img_visenze_cat_map[img_id]
#        print("vi_cats: ", vi_cats)
        ttl_cnt += 1
        if not (len(styles) == len(attrs) and len(attrs) == len(vi_cats)):
            bad_num_cnt += 1
            keep_flag = False
#        print(bad_num_cnt)
        correct_cat_cnt = 0
        correct_idxs = []
        for i, vi_cat in enumerate(vi_cats):
#            print("i: ", i)
#            print("vi_cat: ", vi_cat)
            lo_cat = vi_lo_cat_map[vi_cat]
            if lo_cat == pre_cat:
                correct_cat_cnt += 1
                correct_idxs.append(i)
            else:
               # if lo_cat == "Top":
               #     if data_path == "men" and pre_cat in ["Track Suits", "Coats", "Shirts", "Suits", "T-shirts", "Jackets", "Sweaters"]:
               #         correct_cat_cnt += 1
               #         correct_idxs.append(i)
               #     if data_path == "women" and pre_cat in ["Blazers", "Blouses", "Dresses", "Shirts", "T-shirts", "Sweaters"]:
               #         correct_cat_cnt += 1
               #         correct_idxs.append(i)
               # else:
               continue
        if correct_cat_cnt not in correct_cat_stat:
            correct_cat_stat[correct_cat_cnt] = 1
        else:
            correct_cat_stat[correct_cat_cnt] += 1

        if correct_cat_cnt != 1:
            keep_flag = False

        if "visenz_attrs" in res:
            del item_info[item_id]["visenz_attrs"]
        if "visenze_styles" in res:
            del item_info[item_id]["visenze_styles"]
        if keep_flag:
            i = correct_idxs[0]
            if pre_cat == "Accessories" and pre_cat_2 != "Bags":
                attrs = []
                styles = []
            elif pre_cat == "Footwear":
                attrs = []
                styles = ["::".join([pre_cat,"style", s]) for s in styles[i]]
            elif pre_cat_2 in ["Suits","Track_Suits"]:
                attrs = []
                styles = ["::".join([pre_cat, "style", s]) for s in styles[i]]
            else:
                attrs = []
                for k, v in img_attr_map[img_id][i].items():
                    if v != "na":
                        attrs.append("::".join([pre_cat, k, v]))
                styles = ["::".join([pre_cat, "style", s]) for s in styles[i]]

            #if len(styles)>1:            
            item_info[item_id]["visense_styles"] = styles
            #else:
            #    item_info[item_id]["visense_styles"] = styles
            item_info[item_id]["visense_attrs"] = attrs
            item_info[item_id]["visense_cats"] = img_visenze_cat_map[img_id][i]
        else:
            item_info[item_id]["visense_styles"] = [] 
            item_info[item_id]["visense_attrs"] = []
            item_info[item_id]["visense_cats"] = img_visenze_cat_map[img_id][i]

        tags = ["::".join([pre_cat, tag_type_map[tag], tag]) for tag in res["tags"][1:]]
        item_info[item_id]["new_tags"] = tags

    print("total cnt %d, bad cnt %d" %(ttl_cnt, bad_num_cnt))
    print("how many objects whose category is correctly labelled for each image : the corresponding image numbers ")
    for cat_cnt, cnt in sorted(correct_cat_stat.items(), key=lambda i: i[0], reverse=True):
        print(cat_cnt, cnt)
    print("\n")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    output = data_path + "/item_info_final.json"
    json.dump(item_info, open(output, "w"), indent=4)

    return item_info


def align_taxonomy(data_path):
    attr_set = get_img_attr_set(data_path)

    all_img_ids = set()
    item_info = json.load(open(data_path + "/item_info_new_cat.json"))
    for item_id, res in item_info.items():
        img_id = res["img_id"]
        all_img_ids.add(img_id)

    cat_stat = {}
    for line in open("./items_tagging_res.txt"):
        data = json.loads(line.strip())
        for img_id, res in data.items():
            if img_id not in all_img_ids:
                continue
            for each_res in res["result"]:
                if each_res["tag_group"] == "fashion_attributes":
                    for each_obj in each_res["objects"]:
                        tmp_tags = []
                        for i in each_obj["tags"]:
                            tag_key, tag_value = i["tag"].split(":")
                            if tag_key not in attr_set:
                                tmp_tags.append(i["tag"])
                        tmp_tag = "__".join(tmp_tags)
                        if tmp_tag not in cat_stat:
                            cat_stat[tmp_tag] = 1
                        else:
                            cat_stat[tmp_tag] += 1

    for cat, cnt in sorted(cat_stat.items(), key=lambda i: i[1], reverse=True):
        print("", cat)


def check_pair(h_cat, t_cat):
    query_cat_blacklist = {"Accessories", "Footwear"}
    if h_cat != t_cat and h_cat not in query_cat_blacklist:
        return True
    else:
        return False


def train_test_split(all_pos_pairs):
    random.shuffle(all_pos_pairs) 

    train_len = round(0.7 * len(all_pos_pairs))
    train_test_len = round(0.9 * len(all_pos_pairs))
    train_pairs = all_pos_pairs[:train_len]#70% for training
    test_pairs = all_pos_pairs[train_len:train_test_len]#20% for testing
    val_pairs = all_pos_pairs[train_test_len:]#10% for validation

    return train_pairs, val_pairs, test_pairs


def negtive_sample(all_pos_pairs, pos_pairs, item_info, neg_num, train_or_test, flag):
    all_pos_ids = set()
    for pair in all_pos_pairs:
        h_id, h_cat, t_id, t_cat = pair
        all_pos_ids.add("_".join([h_id, t_id]))

    all_ids = item_info.keys()

    if train_or_test == "train":
        train_pairs = []
        for pos in pos_pairs:
            h_id, h_cat, t_id, t_cat = pos
            neg_ids = []
            candis = random.sample(all_ids, max(neg_num*2, neg_num+100))
            for candi in candis:
                if len(neg_ids) >= neg_num:
                    break
                candi_cat = get_cat(item_info, candi)
                if not flag:
                    if check_pair(h_cat, candi_cat):
                        neg_pair = h_id + "_" + candi
                        if neg_pair not in all_pos_ids:
                            neg_ids.append(candi)
                else:
                    if candi_cat == t_cat:
                        neg_pair = h_id + "_" + candi
                        if neg_pair not in all_pos_ids:
                            neg_ids.append(candi)
            for neg_id in neg_ids:
                train_pairs.append([h_id, t_id, neg_id])

        return train_pairs
    if train_or_test == "test" or train_or_test == "validation":
        test_pairs = []
        test_dict = {}
        for pos in pos_pairs:
            h_id, h_cat, t_id, t_cat = pos
            neg_ids = []
            candis = random.sample(all_ids, max(neg_num*2, neg_num+100))
            for candi in candis:
                if len(neg_ids) >= neg_num:
                    break
                candi_cat = get_cat(item_info, candi)
            #    if check_pair(h_cat, candi_cat):
            #        neg_pair = h_id + "_" + candi
            #        if neg_pair not in all_pos_ids:
            #            neg_ids.append(candi)
                if not flag:
                    if check_pair(h_cat, candi_cat):
                        neg_pair = h_id + "_" + candi
                        if neg_pair not in all_pos_ids:
                            neg_ids.append(candi)
                else:
                    if candi_cat == t_cat:
                        neg_pair = h_id + "_" + candi
                        if neg_pair not in all_pos_ids:
                            neg_ids.append(candi)                

            neg_pairs = []
            for neg_id in neg_ids:
                test_pairs.append([h_id, t_id, neg_id])
                neg_pairs.append(h_id + "_" + neg_id)
            pos_pair = h_id + "_" + t_id
            test_dict[pos_pair] = neg_pairs

        return test_pairs, test_dict


def one_hot_encode(item_info, data_path):
    feature_code = {}
    results = {}
    ori_results = {}

    cnt = 0 
    for item_id, res in item_info.items():
        fea_strs = []
        fea_codes = []
        filtered_fea_strs = []

        cat = res["new_cat"][-1]
        pre_cat = get_cat(item_info, item_id)
        cat = pre_cat + "::category::" + cat
        fea_strs.append(cat)
        if cat not in feature_code:
            feature_code[cat] = cnt
            cnt += 1
        for tag in res["new_tags"]:
            tags = pre_cat + "::" + tag.split("::")[1] + "::" + tag.split("::")[2]
            fea_strs.append(tags)
            if tags not in feature_code:
                feature_code[tags] = cnt
                cnt += 1
        for attr in res["visense_attrs"]:
            if "product_color" in attr or "product_pattern" in attr:
                continue
            else:
                fea_strs.append(attr)
            if attr not in feature_code:
                feature_code[attr] = cnt
                cnt += 1
        for style in res["visense_styles"]:
            fea_strs.append(style)
            if style not in feature_code:
                feature_code[style] = cnt
                cnt += 1
            break
        for fea in fea_strs:
            if fea in feature_code:
                fea_codes.append(feature_code[fea])
                feanew = fea.replace(" ", "_")
                filtered_fea_strs.append(feanew)
        results[item_id] = fea_codes
        ori_results[item_id] = filtered_fea_strs



    print("total number of features is: %d" %(len(feature_code)))
    output = open(data_path + "/feature_map.txt", "w")
    for fea, code in sorted(feature_code.items(), key=lambda i: i[1]):
        output.write(str(code) + "\t" + fea.replace(" ", "_") + "\ti\n")
    output.close()

#    for item_id, res in item_info.items():
#        fea_codes = []
#        filtered_fea_strs = []
#        for fea in fea_strs:
#            fea = fea.replace(" ", "_")
#            if fea in feature_code:
#                fea_codes.append(feature_code[fea])
#                filtered_fea_strs.append(fea)
#        results[item_id] = fea_codes
#        ori_results[item_id] = filtered_fea_strs

    return feature_code, results, ori_results


#def encode_all_items(item_info, fea_code_map):
#    results = {}
#    ori_results = {}
#    for item_id, res in item_info.items():
#        fea_strs = []
#        pre_cat = get_cat(item_info, item_id)
#        fine_category = pre_cat + "::category::" + res["new_cat"][-1]
#        fea_strs.append(fine_category)
#        for tag in res["new_tags"]:
#        #tag = res["new_tags"]
#             tags = pre_cat + "::" + tag.split("::")[1] + "::" + tag.split("::")[2]
#             fea_strs += tags#res["new_tags"]
#        fea_strs += res["visense_styles"]
#        fea_strs += res["visense_attrs"]
#        
#        fea_codes = []
#        filtered_fea_strs = []
#        for fea in fea_strs:
#            fea = fea.replace(" ", "_")
#            if fea in fea_code_map:
#                fea_codes.append(fea_code_map[fea])
#                filtered_fea_strs.append(fea)
#        results[item_id] = fea_codes
#        ori_results[item_id] = filtered_fea_strs
#
#    return results, ori_results


def generate_feature_file(pairs, item_info, filepath, pair_featIdx_map_filepath, data_path_s):
    fea_code_map, item_fea_map, item_orifea_map = one_hot_encode(item_info, data_path_s)

    #item_fea_map, item_orifea_map = encode_all_items(item_info, fea_code_map)
    json.dump(item_orifea_map, open(data_path_s + "/item_feature_map.json", "w"))
    #pause()
    pos_pairs = set()
    res = []
    for pair in pairs:
        h_id, t_id, n_id = pair
        pos_pair = h_id + "_" + t_id
        h_fea = item_fea_map[h_id]
        t_fea = item_fea_map[t_id]
        n_fea = item_fea_map[n_id]
        if pos_pair not in pos_pairs:
            tmp_res = {"pair": pos_pair}
            tmp_res["fea"] = " ".join(["1"] + [str(fea)+":1" for fea in h_fea+t_fea])
            res.append(tmp_res)
            pos_pairs.add(pos_pair)
        neg_pair = h_id + "_" + n_id
        tmp_res = {"pair": neg_pair}
        tmp_res["fea"] = " ".join(["0"] + [str(fea)+":1" for fea in h_fea+n_fea])
        res.append(tmp_res)

    output = open(filepath, "w") 
    random.shuffle(res)
    pair_featIdx_map = {}
    for idx, cont in enumerate(res):
        pair = cont["pair"]
        fea = cont["fea"]
        if pair not in pair_featIdx_map:
            pair_featIdx_map[pair] = idx
        output.write(fea + "\n")
    output.close()
    json.dump(pair_featIdx_map, open(pair_featIdx_map_filepath, "w"))
        

#def generate_train_outfit_pairs_map(outfit_items_map, test_pos_pairs, all_nodes_set, threshhold):
#    test_pos_pairs_set = set()
#    for i in test_pos_pairs:
#        test_pos_pairs_set.add(i[0] + "_" + i[2])
#
#    outfit_len_stat = {}
#    outfit_ttl = len(outfit_items_map) 
#    outfit_over_thr = 0
#    complete_outfit_over_thr = 0
#    train_outfit_pairs_map = {}
#    train_outfit_pairs_len_stat = {}
#    for outfit_id, items in outfit_items_map.items():
#         len_ = len(items)
#         if len_ not in outfit_len_stat:
#             outfit_len_stat[len_] = 1
#         else:
#             outfit_len_stat[len_] += 1
#
#         if len_ < threshhold:
#             continue
#         else:
#             outfit_over_thr += 1
#
#             pos_items_set = set()
#             for i in items:
#                 pos_items_set.add(i)
#
#             pos_node = random.sample(pos_items_set, 1)[0]
#             neg_node = random.sample(all_nodes_set - pos_items_set, 1)[0]
#             neg_items_set = pos_items_set - {pos_node}
#             neg_items_set.add(neg_node)
#
#             pos_pairs = []
#             pos_items_list = list(pos_items_set)
#             flag = False
#             for i, item1 in enumerate(pos_items_list):
#                 for item2 in pos_items_list[i:]:
#                     pair = item1 + "_" + item2
#                     if pair not in test_pos_pairs_set:
#                         pos_pairs.append(pair)
#                     else:
#                         flag = True
#             if not flag:
#                 complete_outfit_over_thr += 1
#             len_ = len(pos_pairs)
#             if len_ not in train_outfit_pairs_len_stat:
#                 train_outfit_pairs_len_stat[len_] = 1
#             else:
#                 train_outfit_pairs_len_stat[len_] += 1
#
#             neg_pairs = []
#             neg_items_list = list(neg_items_set)
#             for i, item1 in enumerate(neg_items_list):
#                 for item2 in neg_items_list[i:]:
#                     pair = item1 + "_" + item2
#                     neg_pairs.append(pair)
#
#             train_outfit_pairs_map[outfit_id] = {"pos": pos_pairs, "neg": neg_pairs}
#
#    for len_, cnt in sorted(outfit_len_stat.items(), key=lambda i: i[0]):
#        print("outfit length: %d, number: %d" %(len_, cnt))
#    print("\noutfit_num_ttl: %d, outfit_num_over_threshhold: %d, complete outfits: %d" %(outfit_ttl, outfit_over_thr, complete_outfit_over_thr))
#
#    print("\n")
#    for len_, cnt in sorted(train_outfit_pairs_len_stat.items(), key=lambda i: i[0]):
#        print("outfit pair length: %d, number: %d" %(len_, cnt))
#
#    return train_outfit_pairs_map
def mkdir(path):

        folder = os.path.exists(path)
        if not folder:
                os.makedirs(path)
                print("---  new folder...  ---")
                print("---  OK  ---")

        else:
                print("---  There is this folder!  ---")



def generate_train_test_set(outfit_file, data_path_f, Ntrain, Ntest, flag, hcat=None, tcat=None):
    item_info = json.load(open(data_path_f + "/item_info_final.json"))
#    fea_code_map, item_fea_map, item_orifea_map = one_hot_encode(item_info, data_path_f)

    #item_fea_map, item_orifea_map = encode_all_items(item_info, fea_code_map)
 #   json.dump(item_orifea_map, open(data_path_f + "/item_feature_map.json", "w"))   
 #   pause()

    if flag:
       print("/n/nUsing type-specific evaluation mode:/n/n")
       print("Head category is ", hcat, "Tail category is ", tcat)
       data_path = data_path_f + "/" + hcat + "_" + tcat
       mkdir(data_path)
    else:
       print("/n/nUsing overall evaluation mode:/n/n")
       data_path = data_path_f + "/allEva"
       mkdir(data_path)

    outfits = json.load(open(outfit_file))
    all_items_set = set()
    all_pos_pairs = []
    outfit_items_map = {}
    for outfit_id, res in outfits.items():
        item_set = []
        for item in res["items"]:
            item_id = item["item_urlmd5"]
            all_items_set.add(item_id)
            pre_cat = get_cat(item_info, item_id)
            item_set.append([item_id, pre_cat])
        outfit_items_map[outfit_id] = [i[0] for i in item_set]

        for i, h in enumerate(item_set):
            for j, t in enumerate(item_set):
                if i != j:
                    h_cat = h[1]
                    t_cat = t[1]
                    if not flag:
                        if check_pair(h_cat, t_cat):
                            all_pos_pairs.append(h+t)
                    else:
                        if h_cat == hcat and t_cat == tcat:
                            all_pos_pairs.append(h+t)
                        else:
                            continue

    print("# split train and test")
    train_pos_pairs, val_pos_pairs, test_pos_pairs = train_test_split(all_pos_pairs)
    print("# negtive sampling")
    train_pairs = negtive_sample(all_pos_pairs, train_pos_pairs, item_info, Ntrain, "train", flag)
    test_pairs, test_dict = negtive_sample(all_pos_pairs, test_pos_pairs, item_info, Ntest, "test", flag)
    val_pairs, val_dict = negtive_sample(all_pos_pairs, val_pos_pairs, item_info, Ntest, "validation", flag)
    print("# generate feature file")

    generate_feature_file(test_pairs, item_info, data_path + "/xgb_test.txt", data_path + "/pair_featIdx_map_test.json", data_path_f)
    json.dump(test_pairs, open(data_path + "/test_ids.json", "w"))
    json.dump(test_dict, open(data_path + "/test_dict.json", "w"))
    print("testing data generation finished")

    generate_feature_file(train_pairs, item_info, data_path + "/xgb_train.txt", data_path + "/pair_featIdx_map_train.json", data_path_f)
    json.dump(train_pairs, open(data_path + "/train_ids.json", "w"))
    print("training data generation finished")
    generate_feature_file(val_pairs, item_info, data_path + "/xgb_val.txt", data_path + "/pair_featIdx_map_val.json", data_path_f)
    json.dump(val_pairs, open(data_path + "/val_ids.json", "w"))
    json.dump(val_dict, open(data_path + "/val_dict.json", "w"))
    print("validation data generation finished")
    #train_outfit_pairs_map = generate_train_outfit_pairs_map(outfit_items_map, test_pos_pairs, all_items_set, 4)
    #json.dump(train_outfit_pairs_map, open(data_path + "/train_outfit_pairs_map.json", "w")) 

    #train_outfits = []
    #for outfit_id, res in train_outfit_pairs_map.items():
    #    train_outfits.append(res)
    #json.dump(train_outfits, open(data_path + "/train_outfits.json", "w")) 


def rel_helper(item_cat_map, train_ids, p_rel_stat, all_rel_stat):
    for q, p, n in train_ids:
        q_cat = item_cat_map[q]
        p_cat = item_cat_map[p]
        n_cat = item_cat_map[n]

        p_rel = "::".join([q_cat, p_cat])
        n_rel = "::".join([q_cat, n_cat])

        if p_rel not in p_rel_stat:
            p_rel_stat[p_rel] = 1
            all_rel_stat[p_rel] = 1
        else:
            p_rel_stat[p_rel] += 1
            all_rel_stat[p_rel] += 1

        if n_rel not in p_rel_stat:
            all_rel_stat[n_rel] = 1
        else:
            all_rel_stat[n_rel] += 1

    return p_rel_stat, all_rel_stat


def stat_rel(data_path, item_cat_map):
    train_ids = json.load(open(data_path + "/train_ids.json"))
    test_ids = json.load(open(data_path + "/test_ids.json"))

    p_rel_stat = {}
    all_rel_stat = {}
    p_rel_stat, all_rel_stat = rel_helper(item_cat_map, train_ids, p_rel_stat, all_rel_stat)
    p_rel_stat, all_rel_stat = rel_helper(item_cat_map, test_ids, p_rel_stat, all_rel_stat)

    output = open(data_path + "/transE_pos_rel_stat.txt", "w")
    for rel, cnt in sorted(p_rel_stat.items(), key=lambda i: i[1], reverse=True):
        output.write("%d %s\n" %(cnt, rel))
    output.close()

    output = open(data_path + "/transE_all_rel_stat.txt", "w")
    rel_output = open(data_path + "/transE_all_rel.txt", "w")
    others_cnt = 0
    for rel, cnt in sorted(all_rel_stat.items(), key=lambda i: i[1], reverse=True):
        if cnt >= 100:
            rel_output.write(rel + "\n")
        output.write("%d %s\n" %(cnt, rel))
    rel_output.write("others\n")
    rel_output.close()
    output.close()


def get_item_cat_map(data_path):
    item_info = json.load(open(data_path + "/item_info_new_cat.json"))

    item_cat_map = {}
    for item_id, res in item_info.items():
        pre_cat = get_transE_cat(item_info, item_id)
        item_cat_map[item_id] = pre_cat

    json.dump(item_cat_map, open(data_path+"/transE_item_cat_map.json", "w"))

    return item_cat_map 


def main():
    #align_taxonomy("men")
#    align_taxonomy("women")
#    pause()
    # to generate the data for sigir19

    #preprocess_metadata("men")
    print("\n\n\ntrain_test generation\n\n\n")
    category = ["Top", "Bottom", "Outerwear", "Fullbody", "Accessories", "Footwear"]
    generate_train_test_set("./men_metadata_new.json", "men", 3, 1000, False, "Fullbody", "Outerwear")

    #generate_train_test_set("./men_metadata_new.json", "men", 3, 1000)
    #print("\n\n\n")
#    print("meta-data processing\n\n\n")
#    preprocess_metadata("women")
#    pause()


    print("\n\n\ntrain_test generation\n\n\n")
    category = ["Top", "Bottom", "Outerwear", "Fullbody", "Accessories", "Footwear"]
    generate_train_test_set("./women_metadata_new.json", "women", 3, 1000, False, "Fullbody", "Outerwear")

    # to generate the data for transE baseline
#    item_cat_map = get_item_cat_map("men")
#    stat_rel("men", item_cat_map)
#    print("\n\n\n")
#    item_cat_map = get_item_cat_map("women")
#    stat_rel("women", item_cat_map)


if __name__ == "__main__":
    main()
