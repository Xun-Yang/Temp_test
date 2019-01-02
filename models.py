import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def _cal_score(h, t, metric):
    score = None
    if metric == "InnerProduct":
        score = h * t

    return torch.sum(score, 1)

def _cal_score_transE(h, t, path, metric):
    score = None
    if metric == "InnerProduct":
        #score =  h*t + 3*(t-h) * path
        score = torch.pow(h + path - t, 2)

    return torch.sum(score, 1)
def BPRLoss(p_score, n_score):
    loss = torch.log(1 + torch.exp((n_score - p_score)))
    loss = torch.sum(loss)
    return loss

def LogLoss(p_score, n_score):
    loss = -torch.log(p_score)-torch.log(1-n_score) 
    loss = torch.sum(loss)
    return loss

def MarginRankingLoss(p_score, n_score, margin):
    criterion = nn.MarginRankingLoss(margin, False).cuda()
    y = Variable(torch.Tensor([-1])).cuda()
    loss = criterion(p_score, n_score, y)
    return loss


class ExpMatch(nn.Module):
    def __init__(self, img_features, meta_fea_len, nhid, metric, loss, tree_pooling, tree_attention, path_encoding, path_initialization, meta_interact_attention, regularization_para, mm_fusion, Use_TEM, all_paths, all_masks, all_Leafnode_weight_mask, item_featureCode_map, item_featureMask_map): 
        super(ExpMatch, self).__init__()
        self.nhid = nhid
        img_fea_len = img_features.shape[1]

        self.imageW = nn.Linear(img_fea_len, nhid).cuda()
        self.imageW.weight.data.normal_(0, 0.01)
       # nn.init.xavier_uniform_(self.imageW.weight.data)
        self.MLP = nn.Linear(2*nhid, nhid).cuda()
        self.MLP.weight.data.normal_(0, 0.005)
        self.mm_fusion = mm_fusion
        self.Use_TEM = Use_TEM


        self.image_embed = nn.Embedding.from_pretrained(img_features, freeze=True).cuda()

        self.meta_embed = nn.Embedding(meta_fea_len, nhid).cuda()
#        self.meta_embed.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform_(self.meta_embed.weight.data)

        intem_num = img_features.shape[0]
        self.item_Fea = nn.Embedding(intem_num, nhid).cuda()
        nn.init.xavier_uniform_(self.item_Fea.weight.data)

        self.path_embed_intialized = nn.Embedding(all_paths.shape[0], nhid).cuda()
        nn.init.xavier_uniform_(self.path_embed_intialized.weight.data)
#        self.path_embed_intialized.weight.data.normal_(0, 0.005)               
        
        self.W_att = nn.Linear(2*nhid, nhid).cuda()
#        self.W_att.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform_(self.W_att.weight.data)
        self.h_att_TEM = nn.Linear(nhid, 1).cuda()
        self.h_att_TEM.weight.data.normal_(0, 0.01)

        self.h_att = nn.Linear(2*nhid, 1).cuda()
        #self.h_att.weight.data = torch.ones(1, 2*nhid)        
        nn.init.constant_(self.h_att.weight.data, 1)
        self.h_meta_interact_att = nn.Linear(nhid, 1).cuda()
        self.h_meta_interact_att.weight.data = torch.ones(1, nhid)        

        self.predict1 = nn.Linear(nhid, 1).cuda()
        nn.init.constant_(self.predict1.weight.data, 1)
        self.predict2 = nn.Linear(nhid, 1).cuda()
        nn.init.constant_(self.predict2.weight.data, 1)

        self.loss = loss
        self.metric = metric
        self.regularization_para = regularization_para

        self.tree_attention = tree_attention #true/false
        self.tree_pooling = tree_pooling #max/average
        self.path_encoding = path_encoding #2-order/average
        self.path_initialization = path_initialization#true/false
        self.meta_interact_attention = meta_interact_attention        

        self.all_paths = Variable(all_paths).cuda()
        self.all_masks = Variable(all_masks).cuda()
        self.all_Leafnode_weight_mask = Variable(all_Leafnode_weight_mask).cuda()
        self.item_featureCode_map = Variable(item_featureCode_map).cuda() 
        self.item_featureMask_map = Variable(item_featureMask_map).cuda()        

    def process_img(self, img_id):
        img_id = torch.squeeze(img_id, 1)
        img_fea = self.image_embed(img_id)
        #res = self.imageW(img_fea)
        item_fea = self.item_Fea(img_id -1)

        res = self.imageW(img_fea)# + item_fea
        res = F.normalize(res, p=2, dim=1)
        return res

    def _cal_score_TEM(self, h, t, path):
        score = None
        if self.metric == "InnerProduct":
            score =  self.predict1(h*t) + 5*self.predict2((t-h)*path)
            #score =  torch.sum((t-h)*path, 1)

        return score

    def process_meta(self, path, mask, user, item):
        path_len = path.shape[2]
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.nhid)
#        self.meta_embed.weight = F.normalize(self.meta_embed.weight, p=2, dim= 3)
        path_embeds = F.normalize(self.meta_embed(path), p=2, dim= 3) * mask
        #path_embeds = F.normalize(path_embeds, p=2, dim= 3) 
        if self.path_encoding == "2-order":
#            path_embeds = path_embeds + torch.ones(path_embeds.shape).cuda() * (1 - mask)
#            path_meta_embeds = path_embeds[:, :, 0::2, :] + mask[:, :, 1::2, :] * path_embeds[:, :, 1::2, :]
            path_meta_embeds = path_embeds[:, :, 0::2, :] + path_embeds[:, :, 1::2, :]#mask[:, :, 1::2, :] * path_embeds[:, :, 1::2, :]
            maxdepth = path_meta_embeds.shape[2]
            #path_res = torch.ones(path_meta_embeds[:, :, 0, :].shape).cuda()
            #for i in range(0, maxdepth):
            #    path_res = path_res * path_meta_embeds[:, :, i, :]


            path_meta_2_order_interact = path_meta_embeds[:, :, 0:(maxdepth-1), :] * path_meta_embeds[:, :, 1:maxdepth, :]
            path_meta_3_order_interact = path_meta_embeds[:, :, 0:(maxdepth-2), :] * path_meta_embeds[:, :, 1:(maxdepth-1), :] * path_meta_embeds[:, :, 2:maxdepth, :]
            path_meta_4_order_interact = path_meta_embeds[:, :, 0:(maxdepth-3), :] * path_meta_embeds[:, :, 1:(maxdepth-2), :] * path_meta_embeds[:, :, 2:(maxdepth-1), :] * path_meta_embeds[:, :, 3:(maxdepth), :]

            if self.meta_interact_attention:
                path_meta_2_order_interact = F.normalize(path_meta_2_order_interact, p=2, dim= 3)
                path_res = self.get_attentive_meta_interact_embedding(path_meta_2_order_interact, user, item)
                path_res = torch.sum(path_res, 2)
            else:
                path_res_1 = F.normalize(torch.mean(path_meta_2_order_interact, 2),p=2, dim=2)
                path_res_2 = F.normalize(torch.mean(path_meta_3_order_interact, 2),p=2, dim=2)
                path_res_3 = F.normalize(torch.mean(path_meta_4_order_interact, 2),p=2, dim=2)
#                path_res = self.W_att(torch.cat((path_res_1,path_res_2),-1))
                path_res = (path_res_1 + path_res_2 + path_res_3)/3
#                path_res = F.normalize(path_res, p=2, dim=2)
        elif self.path_encoding == "average":
            path_meta_embeds = path_embeds[:, :, 0::2, :]
            path_res = torch.mean(path_meta_embeds, 2)# path encoding with meta-data average pooling
            path_res = F.normalize(path_res, p=2, dim=2)

        elif self.path_encoding == "max":
            path_meta_embeds = path_embeds[:, :, 0::2, :]
            path_res = torch.max(path_meta_embeds, 2)[0]#
            path_res = F.normalize(path_res, p=2, dim=2)


        return path_res

    def get_attentive_meta_interact_embedding(self, path_embed, user, item):
        user_item_mul = torch.unsqueeze(user * item, 1).expand([path_embed.shape[0], path_embed.shape[1], path_embed.shape[3]])
        user_item_mul = torch.unsqueeze(user_item_mul, 2).expand(path_embed.shape)
        user_item_sub = torch.unsqueeze(user - item, 1).expand([path_embed.shape[0], path_embed.shape[1], path_embed.shape[3]])
        user_item_sub = torch.unsqueeze(user_item_sub, 2).expand(path_embed.shape)
        fusion = user_item_mul - user_item_sub * path_embed

        weight = torch.squeeze(self.h_meta_interact_att(fusion), -1)
        weight = F.softmax(weight/0.5, dim=-1)
        att_weight = torch.unsqueeze(weight, -1).expand(path_embed.shape)
        res = path_embed * att_weight
        return res



    def get_pooled_path_embedding(self, path_embed, user, item, leafnodeMask = None):
 #       path_embed = F.normalize(path_embed, p=2, dim=2) # path-encoding L2-normalization
      
        if self.tree_attention:#true
            if not self.Use_TEM:
                 user_item_mul = torch.unsqueeze(user * item, 1).expand(path_embed.shape)
                 user_item_sub = torch.unsqueeze(user - item, 1).expand(path_embed.shape)
                 fusion = torch.cat((-user_item_sub * path_embed, user_item_mul),-1)  
                 weight = torch.squeeze(self.h_att(fusion), -1)
         #       weight = weight * F.sigmoid(leafnodeMask * 2)
                 weight = F.softmax(weight/0.2, dim=-1)
                 #print("attention: ", weight[0][0:50])
                 att_weight = torch.unsqueeze(weight, -1).expand(path_embed.shape)
                 res = path_embed * att_weight
            else:
                 user_item_mul = torch.unsqueeze(user * item, 1).expand(path_embed.shape)
                 fusion = torch.cat((user_item_mul, path_embed),-1) 
                 weight = torch.squeeze(self.h_att_TEM(F.relu(self.W_att(fusion))), -1)
                 weight = F.softmax(weight/1, dim=-1)
                 att_weight = torch.unsqueeze(weight, -1).expand(path_embed.shape)
                 res = path_embed * att_weight
        else:
            res = path_embed

        if self.tree_pooling == "max":
            res = torch.max(res, 1)[0]
 #           print("Max-pooling applied")
        elif self.tree_pooling == "average" and self.tree_attention:# attention+ average pooling
            res = torch.sum(res, 1)
 #           print("attention is used. tree pooling strategy is Average-Polling")
        else:                    # no attention + average pooling
            res = torch.mean(res, 1)
 #           print("attention is not used. tree pooling strategy is Average-Polling")
 #       pause()
        return res#, weight

    def generate_all_path_embed(self):
        path_len = self.all_paths.shape[1]
        mask = self.all_masks.unsqueeze(-1).expand(-1, -1, self.nhid)
        #self.meta_embed.weight = F.normalize(self.meta_embed.weight, p=2, dim= 2)
        path_embeds = F.normalize(self.meta_embed(self.all_paths), p=2, dim= 2) * mask
#        path_embeds = F.normalize(path_embeds, p=2, dim= 2)

        if self.path_encoding == "2-order":
 #           path_embeds = path_embeds + torch.ones(path_embeds.shape).cuda() * (1 - mask)
            path_meta_embeds = path_embeds[:, 0::2, :] + path_embeds[:, 1::2, :]#mask[:, 1::2, :] * path_embeds[:, 1::2, :]
 #           path_meta_embeds = path_embeds[:, 0::2, :] + mask[:, 1::2, :] * path_embeds[:, 1::2, :]
            maxdepth = path_meta_embeds.shape[1]
            #path_res = torch.ones(path_meta_embeds[:, 0, :].shape).cuda()
            #for i in range(0, maxdepth):
            #    path_res = path_res * path_meta_embeds[:, i, :]

            path_meta_2_order_interact = path_meta_embeds[:, 0:(maxdepth-1), :] * path_meta_embeds[:, 1:maxdepth, :]
            path_meta_3_order_interact = path_meta_embeds[:, 0:(maxdepth-2), :] * path_meta_embeds[:, 1:(maxdepth-1), :] * path_meta_embeds[:, 2:maxdepth, :]
            path_meta_4_order_interact = path_meta_embeds[:, 0:(maxdepth-3), :] * path_meta_embeds[:, 1:(maxdepth-2), :] * path_meta_embeds[:, 2:(maxdepth-1), :] * path_meta_embeds[:, 3:(maxdepth), :]
            #path_res = torch.mean(path_meta_2_order_interact, 1) + torch.mean(path_meta_3_order_interact, 1)
            path_res_1 = F.normalize(torch.mean(path_meta_2_order_interact, 1), p=2, dim=1) 
            path_res_2 = F.normalize(torch.mean(path_meta_3_order_interact, 1), p=2, dim=1)
            path_res_3 = F.normalize(torch.mean(path_meta_4_order_interact, 1), p=2, dim=1)
            path_res = (path_res_1 + path_res_2 + path_res_3)/3
 #           path_res = F.normalize(path_res, p=2, dim=1)
        #else:
        #    path_res = torch.mean(path_embeds, 1)# path encoding with meta-data average pooling
        elif self.path_encoding == "average":
            path_meta_embeds = path_embeds[:, 0::2, :]
            path_res = torch.mean(path_meta_embeds, 1)# path encoding with meta-data average pooling
            path_res = F.normalize(path_res, p=2, dim=1)
        elif self.path_encoding == "max":
            path_meta_embeds = path_embeds[:, 0::2, :]
            path_res = torch.max(path_meta_embeds, 1)[0]#
            path_res = F.normalize(path_res, p=2, dim=1)

        return path_res 


    def compute_pairwise_score(self, qry_id, res_id, path_ids, all_path_embed=None, path=None, mask=None):

        qry_img_fea = self.process_img(qry_id)
        res_img_fea = self.process_img(res_id)
        if self.mm_fusion:
            qry_meta_fea = self.process_item_meta_fea(qry_id)
            res_meta_fea = self.process_item_meta_fea(res_id)
            qry_fea = qry_img_fea + qry_meta_fea
            res_fea = res_meta_fea + res_img_fea
#            qry_fea = self.MLP(torch.cat((qry_img_fea, qry_meta_fea), 1))
#            res_fea = self.MLP(torch.cat((res_img_fea, res_meta_fea), 1))
        else:
            qry_fea = qry_img_fea
            res_fea = res_img_fea

        if self.path_initialization:
            path_embed = self.path_embed_intialized(path_ids)
            #if not self.Use_TEM:
            path_embed = F.normalize(path_embed, p=2, dim= 2)
        elif self.meta_interact_attention:
            path_embed = self.process_meta(path, mask, qry_fea, res_fea)
        else: 
            path_embed = all_path_embed[path_ids]


        leafnodemask = self.all_Leafnode_weight_mask[path_ids]
        pooled_path_embed = self.get_pooled_path_embedding(path_embed, qry_fea, res_fea, leafnodemask)
        if not self.Use_TEM:
            #score = _cal_score_transE(qry_fea, res_fea, pooled_path_embed, self.metric)
            score = self._cal_score_TEM(qry_fea, res_fea, pooled_path_embed)
        else:
            score = self._cal_score_TEM(qry_fea, res_fea, pooled_path_embed)
        
        return score   #.sigmoid(self.predict(pooled_path_embed)) #+ score

    def process_item_meta_fea(self, item_id):
         item_idx = item_id.squeeze(-1) - 1
         item_meta_code = self.item_featureCode_map[item_idx]
         
         item_meta_mask = self.item_featureMask_map[item_idx]
         item_meta_num = torch.sum(item_meta_mask, 1)


#         meta_Num = self.item_featureCode_map.shape[1]
         item_embeds = self.meta_embed(item_meta_code)
         mask = item_meta_mask.unsqueeze(-1).expand(item_embeds.shape)
         item_embeds = item_embeds * mask
         res = torch.sum(item_embeds, 1)/(item_meta_num.unsqueeze(-1).expand(-1, self.nhid))
         #res = F.normalize(res, p=2, dim=1)

         return res
    def forward(self, qry_id, pos_id, neg_id, pos_path, pos_mask, pos_path_ids, neg_path, neg_mask, neg_path_ids, pos_outfit_ids=None, neg_outfit_ids=None):

        qry_img_fea = self.process_img(qry_id)
        pos_img_fea = self.process_img(pos_id)
        neg_img_fea = self.process_img(neg_id)
        if self.mm_fusion:
            qry_meta_fea = self.process_item_meta_fea(qry_id)
            qry_fea = qry_img_fea + qry_meta_fea
       
            pos_meta_fea = self.process_item_meta_fea(pos_id)
            pos_fea = pos_meta_fea + pos_img_fea
            neg_meta_fea = self.process_item_meta_fea(neg_id)
            neg_fea = neg_meta_fea + neg_img_fea
        
        else:
            qry_fea = qry_img_fea
            pos_fea = pos_img_fea
            neg_fea = neg_img_fea
        
        #VSE_loss = (qry_img_fea - qry_meta_fea).norm(2) + (pos_img_fea - pos_meta_fea).norm(2) + (neg_img_fea - neg_meta_fea).norm(2)

        if self.path_initialization:
#            print(" path embeddings are initialized randomly")
            p_path_embed = self.path_embed_intialized(pos_path_ids)
            n_path_embed = self.path_embed_intialized(neg_path_ids)
            #if not self.Use_TEM:
            p_path_embed = F.normalize(p_path_embed, p=2, dim= 2)
            n_path_embed = F.normalize(n_path_embed, p=2, dim= 2)
           # Embed_L2Norm = self.path_embed_intialized.weight.norm(2) + qry_img_fea.norm(2) + pos_img_fea.norm(2) + neg_img_fea.norm(2)
            Embed_L2Norm =  qry_fea.norm(2) + pos_fea.norm(2) + neg_fea.norm(2)# + self.path_embed_intialized.weight.norm(2)
        else: 
#            print("metadata-wise path encoding computation")

            p_path_embed = self.process_meta(pos_path, pos_mask, qry_fea, pos_fea)
            n_path_embed = self.process_meta(neg_path, neg_mask, qry_fea, neg_fea)
            Embed_L2Norm = self.meta_embed.weight.norm(2) + qry_fea.norm(2) + pos_fea.norm(2) + neg_fea.norm(2)
            
        p_pooled_path_embed = self.get_pooled_path_embedding(p_path_embed,  qry_fea, pos_fea, None)
        n_pooled_path_embed  = self.get_pooled_path_embedding(n_path_embed, qry_fea, neg_fea, None)
        if not self.Use_TEM:        
            #p_score = _cal_score_transE(qry_fea, pos_fea, p_pooled_path_embed, self.metric)
            #n_score = _cal_score_transE(qry_fea, neg_fea, n_pooled_path_embed, self.metric)
            p_score = self._cal_score_TEM(qry_fea, pos_fea, p_pooled_path_embed)
            n_score = self._cal_score_TEM(qry_fea, neg_fea, n_pooled_path_embed)           
        else:
            p_score = self._cal_score_TEM(qry_fea, pos_fea, p_pooled_path_embed)
            n_score = self._cal_score_TEM(qry_fea, neg_fea, n_pooled_path_embed)
        #p_score = _cal_score(qry_img_fea, pos_img_fea, self.metric)   
        #n_score = _cal_score(qry_img_fea, neg_img_fea, self.metric) 


        if self.loss == "BPRLoss":
            pair_loss = BPRLoss(p_score, n_score)
            #pair_loss = MarginRankingLoss(p_score, n_score, 1.0)

        ### outfit related
        if pos_outfit_ids is not None and neg_outfit_ids is not None:
            pos_outfit_embed = self.process_img(pos_outfit_ids[:, :, 0]) * self.process_img(pos_outfit_ids[:, :, 1])
            pos_outfit_embed = torch.mean(pos_outfit_embed, 1)
            pos_outfit_score = torch.sum(pos_outfit_embed, 1)

            neg_outfit_embed = self.process_img(neg_outfit_ids[:, :, 0]) * self.process_img(neg_outfit_ids[:, :, 1])
            neg_outfit_embed = torch.mean(neg_outfit_embed, 1)
            neg_outfit_score = torch.sum(neg_outfit_embed, 1)

            if self.loss == "BPRLoss":
                outfit_loss = BPRLoss(pos_outfit_score, neg_outfit_score)

            return pair_loss, outfit_loss

        return pair_loss + self.regularization_para * Embed_L2Norm #+ self.regularization_para * VSE_loss
