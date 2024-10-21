import torch
from torch import nn
from einops import rearrange

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, drop_p):
        super().__init__()
        
        self.self_atten = nn.MultiheadAttention(d_model, n_heads, drop_p, batch_first = True) 
        self.self_atten_LN = nn.LayerNorm(d_model)
        
        self.enc_dec_atten = nn.MultiheadAttention(d_model, n_heads, drop_p, batch_first = True)
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(2 * d_model, d_model),
        )
        self.ff_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):

        residual, _ = self.self_atten(x, x, x, attn_mask = dec_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual, _ = self.enc_dec_atten(x, enc_out, enc_out, attn_mask = enc_dec_mask)
        residual = self.dropout(residual)
        x = self.enc_dec_atten_LN(x + residual)

        residual = self.ff(x)
        residual = self.dropout(residual)
        x = self.ff_LN(x + residual)

        return x
    

class Captioner(nn.Module):
    def __init__(self, feature_extractor, feature_channels, num_layers, d_model, max_len, n_heads, vocab_size, drop_p, pad_idx, device):
        super().__init__()

        self.pad_idx = pad_idx
        self.device = device
        
        self.n_heads = n_heads

        self.feature_extractor = feature_extractor
        # self.feature_extractor.eval()
        self.feature_projection = nn.Linear(feature_channels, d_model)
        
        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, drop_p) for _  in range(num_layers)]
        )

        self.dropout = nn.Dropout(drop_p)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Weight initialization
        after_embedding = False
        for m in self.modules():
            if hasattr(m, "weight"):
                if isinstance(m, nn.Embedding):
                    after_embedding = True
                if after_embedding and m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight)

    def make_dec_mask(self, trg):
        trg_pad_mask = (trg.to("cpu") == self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1) #개헤단단
        trg_future_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1])) == 0
        dec_mask = trg_pad_mask | trg_future_mask
        dec_mask = rearrange(dec_mask, "개 헤 단 차 -> (개 헤) 단 차")
        return dec_mask

    def forward(self, img, txt):

        assert img.shape[1] == 3
        img = self.feature_extractor(img)
        
        img = rearrange(img, "B C H W -> B (H W) C")
        img = self.feature_projection(img)
        
        dec_mask = self.make_dec_mask(txt)#.to(self.device)
        enc_dec_mask = None # img로 부터 뽑은 feature 이므로

        pos = torch.arange(txt.shape[1]).repeat(txt.shape[0], 1)#.to(self.device)
        txt = self.scale * self.input_embedding(txt) + self.pos_embedding(pos)
        txt = self.dropout(txt)

        for layer in self.layers:
            x = layer(txt, img, dec_mask, enc_dec_mask)
        
        x = self.fc_out(x)
        
        return x