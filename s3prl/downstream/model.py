import torch
import torch.nn as nn
import torch.nn.functional as F


def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model


class FrameLevel(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, activation='ReLU', **kwargs):
        super().__init__()
        latest_dim = input_dim
        self.hiddens = []
        if hiddens is not None:
            for dim in hiddens:
                self.hiddens += [
                    nn.Linear(latest_dim, dim),
                    getattr(nn, activation)(),
                ]
                latest_dim = dim
        self.hiddens = nn.Sequential(*self.hiddens)
        self.linear = nn.Linear(latest_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        hidden_states = self.hiddens(hidden_state)
        logit = self.linear(hidden_state)

        return logit, features_len


class UtteranceLevel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        latest_dim = input_dim
        self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation)
        post_net_conf = post_net.get(post_net['select'], {})
        if 'input_dim' in post_net_conf:
            latest_dim = post_net_conf['input_dim']
            post_net[post_net['select']].pop('input_dim')
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)

    def forward(self, hidden_state, features_len=None):
        if self.pre_net is not None:
            hidden_state, features_len = self.pre_net(hidden_state, features_len)

        if type(self.pooling).__name__ == 'CP':
            pooled  = self.pooling(hidden_state, features_len)
        else:
            pooled, features_len = self.pooling(hidden_state, features_len)
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len


class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()


class MeanStdPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanStdPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        '''
       	Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape
            features_len  - [B] of feature length
       	'''
	agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            mean_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0) # (T', H) -> (H,)
            std_vec  = torch.std(feature_BxTxH[i][:features_len[i]], dim=0)  # (H,)
            agg_vec  = torch.cat((mean_vec, std_vec), dim=0) # (2H,)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()


class CP(nn.Module):
    '''

    Correlation type of pooling: https://arxiv.org/abs/2104.02571

    '''

    def __init__(self, out_dim=0, input_dim=0, p_drop=0.25, **kwargs):
        super(CP, self).__init__()
        # input_dim, out_dim not used
        self.drop_f = p_drop != 0
        if self.drop_f:
            self.dropout = nn.Dropout2d(p=p_drop)

    def forward(self, feature_BxTxH, att_mask_BxT, **kwargs):
       	'''
        Arguments
            feature - [BxTxH]   Acoustic feature with shape
            att_mask- [BxT]     Attention Mask logits
       	'''
        eps = 1e-10
        if self.drop_f:
            feature_BxTxH = torch.permute(feature_BxTxH, (0,2,1)) # [BxHxT]
            feature_BxTxH = torch.unsqueeze(feature_BxTxH, dim=-1) # [BxHxTx1]
            feature_BxTxH = self.dropout(feature_BxTxH)
            feature_BxTxH = torch.squeeze(feature_BxTxH, dim=-1) # [BxHxT]
            feature_BxTxH = torch.permute(feature_BxTxH, (0,2,1)) # [BxTxH]
            feature_BxTxH += eps

        # note: do not include diagonal
       	dshift = 1  # the diagonal to consider (0:includes diag, 1:from 1 over diag)
       	agg_vec_list = []
       	for i in range(len(feature_BxTxH)):
            if torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False).size(0) == 0:
               	length = len(feature_BxTxH[i])
            else:
               	length = torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False)[0] + 1

	    x = feature_BxTxH[i][:length] # (T', H)
            # normalization
            x = torch.div((x - torch.mean(x, dim=0, keepdim=True)), torch.std(x, dim=0, keepdim=True)+1e-9)
            corr = torch.div(torch.einsum('jk,jl->kl', x, x), x.shape[0]) # (H, H)
            # select upper triangular matrix, vectorize
            corr = corr[torch.triu_indices(corr.shape[0], corr.shape[1], offset=dshift).unbind()]
            agg_vec_list.append(corr)

       	return torch.stack(agg_vec_list) # (B,feat_dim)


class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim, activation, **kwargs):
        super(AttentivePooling, self).__init__()
        self.sap_layer = AttentivePoolingModule(input_dim, activation)

    def forward(self, feature_BxTxH, features_len):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)

        return sap_vec, torch.ones(len(feature_BxTxH)).long()


class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w


class CorrelationAttentivePooling(nn.Module):
    '''

    Correlation type of pooling: https://arxiv.org/abs/2104.02571
    Correlation attentive pooling: https://arxiv.org/abs/2211.01756

    '''

    def __init__(self, p_drop=0.25, **kwargs):
       	super(CorrelationAttentivePooling, self).__init__()
       	self.nheads = 4
       	input_dim = 256
       	self.sap_layer = AttentivePoolingModule(input_dim=input_dim) if self.nheads==1 \
                                                           else AttentiveLSEPoolingModule(input_dim=input_dim, nheads=self.nheads)
        self.drop_f = p_drop != 0
        print(f'\nSelected channel wise dropout rate of: {p_drop}\n')
        print(f'\nSelected number of attention heads: {self.nheads}\n')
        if self.drop_f:
            self.dropout = nn.Dropout2d(p=p_drop)

    def forward(self, feature_BxTxH, features_len, **kwargs):
        '''
       	Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape
            features_len  - [B] of feature length                                                                                        \

       	'''

	if self.drop_f:
            feature_BxHxT = torch.permute(feature_BxTxH, (0,2,1)) # [BxHxT]
            feature_BxHxT = torch.unsqueeze(feature_BxHxT, dim=-1) # [BxHxTx1]
            feature_BxHxT = self.dropout(feature_BxHxT)
            feature_BxHxT = torch.squeeze(feature_BxHxT, dim=-1) # [BxHxT]
            feature_BxTxH = torch.permute(feature_BxHxT, (0,2,1)) # [BxTxH]
        
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        if self.nheads == 1:
            att_logits_before_lse = []
            sap_vec, att_w = self.sap_layer(feature_BxTxH, len_masks)
        else:
            sap_vec, att_w, att_logits_before_lse = self.sap_layer(feature_BxTxH, len_masks)

        dshift = 1  # the diagonal to consider (0:includes diag, 1:from 1 over diag)

        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            d = feature_BxTxH.shape[-1]
            x = feature_BxTxH[i] - sap_vec[i]
            corr = torch.einsum('jk,jl->kl', att_w[i]*x, x) # (H, H)
            if dshift == 1:
                s = torch.sqrt(torch.diagonal(corr)+1e-9)
                corr = torch.div(corr,torch.outer(s, s))
            corr = corr[torch.triu_indices(d, d, offset=dshift).unbind()]
            agg_vec_list.append(corr)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long() #, att_w, att_logits_before_lse # (B,feat_dim), (B,)


class AttentiveLSEPoolingModule(nn.Module):
    """

    Implementation of LSE Attentive Pooling

    """
    def __init__(self, input_dim, nheads, activation='ReLU', **kwargs):
        super(AttentiveLSEPoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, nheads)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
       	"""
       	input:                                                                                                                                                $
       	batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension                                                                    $
       	attention_weight:                                                                                                                                     $
       	att_w : size (B, T, 1)
       	return:                                                                                                                                               $
       	utter_rep: size (B, H)
       	"""

        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits_before_lse = att_logits
        att_logits = torch.logsumexp(att_logits, dim=-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

       	return utter_rep, att_w, att_logits_before_lse
