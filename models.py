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
        score =  h*t + (t-h) * path

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
    def __init__(self, img_features, meta_fea_len, nhid, metric, loss, tree_pooling, tree_attention, path_encoding, regularization_para, all_paths, all_masks, all_Leafnode_weight_mask): 
        super(ExpMatch, self).__init__()
        self.nhid = nhid
        img_fea_len = img_features.shape[1]

        self.imageW = nn.Linear(img_fea_len, nhid).cuda()
        self.imageW.weight.data.normal_(0, 0.005)

        self.image_embed = nn.Embedding.from_pretrained(img_features, freeze=True).cuda()

        self.meta_embed = nn.Embedding(meta_fea_len, nhid).cuda()
#        self.meta_embed.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform_(self.meta_embed.weight.data)
        
        self.W_att = nn.Linear(nhid, nhid).cuda()
        self.W_att.weight.data.normal_(0, 0.05)

        self.h_att = nn.Linear(nhid, 1).cuda()
        self.h_att.weight.data = torch.ones(1, nhid)        

        self.predict = nn.Linear(nhid, 1).cuda()
        self.predict.weight.data = torch.ones(1, nhid)
#        self.predict.weight.data.normal_(0, 0.01)
        self.loss = loss
        self.metric = metric
        self.regularization_para = regularization_para

        self.tree_attention = tree_attention #true/false
        self.tree_pooling = tree_pooling #max/average
        self.path_encoding = path_encoding #2-order/average

        self.all_paths = Variable(all_paths).cuda()
        self.all_masks = Variable(all_masks).cuda()
        self.all_Leafnode_weight_mask = Variable(all_Leafnode_weight_mask).cuda()
        

    def process_img(self, img_id):
        img_id = torch.squeeze(img_id, 1)
        img_fea = self.image_embed(img_id)
        res = self.imageW(img_fea)
      #  res = F.normalize(res, p=2, dim=1)
        return res


    def process_meta(self, path, mask):
        path_len = path.shape[2]
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.nhid)
        path_embeds = self.meta_embed(path) * mask
        
        if self.path_encoding == "2-order":
            path_embeds = path_embeds + torch.ones(path_embeds.shape).cuda() * (1 - mask) 
            meta_0 = path_embeds[:, :, 0, :] + mask[:, :, 1, :] * path_embeds[:, :, 1, :]
            meta_1 = path_embeds[:, :, 2, :] + mask[:, :, 3, :] * path_embeds[:, :, 3, :]
            path_res = meta_0 * meta_1
            
            for i in range(1, int(path_len/2)-1):
                meta_i   = path_embeds[:, :, i*2, :] + mask[:, :, i*2 +1, :] * path_embeds[:, :, i*2 +1, :]
                meta_i_1 = path_embeds[:, :, 2*(i+1), :] + mask[:, :, 2*(i+1)+1, :] * path_embeds[:, :, 2*(i+1)+1, :]
                path_res = path_res + meta_i * meta_i_1
            
        else:
            path_res = torch.mean(path_embeds, 2)# path encoding with meta-data average pooling

        path_res = F.normalize(path_res, p=2, dim=2) # path-encoding L2-normalization

        return path_res


    def get_pooled_path_embedding(self, path_embed, leafnodeMask, user, item):
        
#        leafnodeMask = torch.unsqueeze(leafnodeMask, 2).expand(path_embed.shape) 
#        path_embed = leafnodeMask * path_embed
        
        if self.tree_attention:#true
            #user_item = user * item
            user_item_mul = torch.unsqueeze(user * item, 1).expand(path_embed.shape)
            user_item_sub = torch.unsqueeze(user - item, 1).expand(path_embed.shape)
            #concat = torch.cat([path_embed, user_item_expand], dim=-1)
            fusion = user_item_mul - user_item_sub * path_embed
            weight = torch.squeeze(self.h_att(fusion), -1)
            
            weight = weight * F.sigmoid(leafnodeMask * 2)
            
           # print(weight[0][0:9])
            
            weight = F.softmax(weight, dim=-1)
            #print("attention: ", weight[0][0:50])
            #pause()
#            weight = F.relu(self.W_att(concat))
#            weight = self.h_att(weight)
#            weight = F.softmax(weight, dim=-1)
            att_weight = torch.unsqueeze(weight, -1).expand(path_embed.shape)
            res = path_embed * att_weight
        else:
            res = path_embed

        if self.tree_pooling == "max":
            res = torch.max(res, 1)[0]
        elif self.tree_attention:# attention+ average pooling
            res = torch.sum(res, 1)
        else:                    # no attention + average pooling
            res = torch.mean(res, 1)

    
        return res#, weight


    def generate_all_path_embed(self):
        path_len = self.all_paths.shape[1]
        mask = self.all_masks.unsqueeze(-1).expand(-1, -1, self.nhid)
        path_embeds = self.meta_embed(self.all_paths) * mask

        if self.path_encoding == "2-order":
            path_embeds = path_embeds + torch.ones(path_embeds.shape).cuda() * (1 - mask)
            meta_0 = path_embeds[:, 0, :] + mask[:, 1, :] * path_embeds[:, 1, :]
            meta_1 = path_embeds[:, 2, :] + mask[:, 3, :] * path_embeds[:, 3, :]
            path_res = meta_0 * meta_1

            for i in range(1, int(path_len/2)-1):
                meta_i   = path_embeds[:, i*2, :] + mask[:, i*2 +1, :] * path_embeds[:, i*2 +1, :]
                meta_i_1 = path_embeds[:, 2*(i+1), :] + mask[:, 2*(i+1)+1, :] * path_embeds[:, 2*(i+1)+1, :]
                path_res = path_res + meta_i * meta_i_1

        else:
            path_res = torch.mean(path_embeds, 1)# path encoding with meta-data average pooling

        path_res = F.normalize(path_res, p=2, dim=1)

        return path_res 


    def compute_pairwise_score(self, qry_id, res_id, path_ids, all_path_embed):

        qry_img_fea = self.process_img(qry_id)
        res_img_fea = self.process_img(res_id)

        path_embed = all_path_embed[path_ids]
        leafnodemask = self.all_Leafnode_weight_mask[path_ids]
        pooled_path_embed = self.get_pooled_path_embedding(path_embed, leafnodemask, qry_img_fea, res_img_fea)

        score = _cal_score_transE(qry_img_fea, res_img_fea, pooled_path_embed, self.metric)
        
        return score   #.sigmoid(self.predict(pooled_path_embed)) #+ score


    def forward(self, qry_id, pos_id, neg_id, pos_path, pos_mask, pos_leafnodeMask, neg_path, neg_mask, neg_leafnodeMask, pos_outfit_ids=None, neg_outfit_ids=None):

        qry_img_fea = self.process_img(qry_id)
        pos_img_fea = self.process_img(pos_id)
        neg_img_fea = self.process_img(neg_id)

        p_path_embed = self.process_meta(pos_path, pos_mask)
        p_pooled_path_embed = self.get_pooled_path_embedding(p_path_embed, pos_leafnodeMask, qry_img_fea, pos_img_fea)


        n_path_embed = self.process_meta(neg_path, neg_mask)
        n_pooled_path_embed  = self.get_pooled_path_embedding(n_path_embed, neg_leafnodeMask, qry_img_fea, neg_img_fea)

        Embed_L2Norm = self.meta_embed.weight.norm(2) + qry_img_fea.norm(2) + pos_img_fea.norm(2) + neg_img_fea.norm(2)
        
        p_score = _cal_score_transE(qry_img_fea, pos_img_fea, p_pooled_path_embed, self.metric)
        n_score = _cal_score_transE(qry_img_fea, neg_img_fea, n_pooled_path_embed, self.metric)

        #p_score = _cal_score(qry_img_fea, pos_img_fea, self.metric)   
        #n_score = _cal_score(qry_img_fea, neg_img_fea, self.metric) 


        if self.loss == "BPRLoss":
            pair_loss = BPRLoss(p_score, n_score)

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

        return pair_loss + self.regularization_para * Embed_L2Norm
