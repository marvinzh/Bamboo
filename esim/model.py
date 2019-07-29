import torch

class ESIM(torch.nn.Module):
    def __init__(self, n_vocab, d_embed, d_hidden, n_layers, d_proj, d_v, n_layers_cmp, d_pred, dropout_rate=0.):
        super().__init__()
        # input encoding layer
        self.embed = torch.nn.Embedding(n_vocab, d_embed)
        self.in_enc_pre = torch.nn.LSTM(d_embed, d_hidden, n_layers, dropout=dropout_rate, bidirectional=True)
        self.in_enc_hypo = torch.nn.LSTM(d_embed, d_hidden, n_layers, dropout=dropout_rate, bidirectional=True)
        
        # inference composition
        self.proj = torch.nn.Linear(d_hidden*4*2, d_proj)
        self.cmp_pre = torch.nn.LSTM(d_proj, d_v, n_layers_cmp, bidirectional=True)
        self.cmp_hypo = torch.nn.LSTM(d_proj, d_v, n_layers_cmp, bidirectional=True)
        
        # prediction
        self.pred_hidden = torch.nn.Linear(d_v*4*2, d_pred)
        self.out = torch.nn.Linear(d_pred, 3)
    
    def intra_sent_att(self, sent1, sent2):
        # sent1 (L1, B, H)
        # sent2 (L2, B, H)
        # return attended representation for sent1 (L1, B, H)
        
        # (B, L1, H)
        sent1 = sent1.permute(1, 0, 2)
        # (B, H, L2)
        sent2 = sent2.permute(1, 2, 0)
        
        # (B, L1, L2)
        score = torch.bmm(sent1, sent2)
        weights = torch.softmax(score, dim=2)
        
        attends=[]
        for w, b in zip(weights, sent2):
            w = w.unsqueeze(1)
            b = b.permute(1,0).unsqueeze(0).repeat(w.size(0), 1, 1)
            attended = torch.bmm(w,b).squeeze(1)
            attends.append(attended)
        
        attended_a = torch.stack(attends)
#         print(attended_a.shape, sent1.shape)
        return attended_a.permute(1,0,2)
    
    def forward(self, premise, hypos):
        pre_embeded = self.embed(premise)
        hypos_embeded =self.embed(hypos)
        
        # enhanced feature
        a_bar,(_,_) = self.in_enc_pre(pre_embeded)
        b_bar,(_,_) = self.in_enc_hypo(hypos_embeded)
        
        a_wave = self.intra_sent_att(a_bar, b_bar)
        b_wave = self.intra_sent_att(b_bar, a_bar)
        
        a_diff = a_bar - a_wave 
        b_diff = b_bar - b_wave
        
        a_prod = a_bar * a_wave
        b_prod = b_bar * b_wave
        
        m_a = torch.cat([a_bar, a_wave, a_diff, a_prod], dim=-1)
        m_b = torch.cat([b_bar, b_wave, b_diff, b_prod], dim=-1)
        
        # feature compostion
        proj_a = torch.relu(self.proj(m_a))
        proj_b = torch.relu(self.proj(m_b))
        
        # (L, B ,H)
        v_a,(_,_) = self.cmp_pre(proj_a)
        v_b,(_,_) = self.cmp_hypo(proj_b)
        
        avg_a = torch.mean(v_a, dim=0) / v_a.size(0)
        max_a,_ = torch.max(v_a, dim=0)
        avg_b = torch.mean(v_b, dim=0) / v_b.size(0)
        max_b,_ = torch.max(v_b, dim=0)
        
        v = torch.cat([avg_a, max_a, avg_b, max_b], dim=-1)
        
        # MLP classifier
        hid = torch.tanh(self.pred_hidden(v))
        logits = self.out(hid)
        
        return logits