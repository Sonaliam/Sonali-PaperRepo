class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=38):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hid_dim)
    def forward(self, trg, enc_src, src, trg_mask, src_mask, imgs, ent, gent):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        index1 = src
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        trg = self.tok_embedding(trg)
        trg = trg * self.scale
        t = self.pos_embedding(pos)
        trg = self.dropout( trg  + t)
        trg1 = trg

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask, imgs, ent, gent)
        output = self.fc_out(trg)

        return output


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.mask_MA = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.msa = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msa1 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msa2 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msa3 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.positionwise_feedforward1 = PositionwiseFeedforwardLayer(hid_dim,                                                                   pf_dim,                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.v1 = nn.Linear(hid_dim, hid_dim)
        self.v2 = nn.Linear(hid_dim, hid_dim)
        self.GELU = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, trg, enc_src, trg_mask, src_mask, img, ent, gent):
        c_t, _ = self.mask_MA(trg, trg, trg, trg_mask)
        c_t = self.self_attn_layer_norm(trg + self.dropout(c_t))
        img, attention = self.msa(c_t, img, img, None)
        enc_src, attention = self.msa1(c_t, enc_src, enc_src, None)
        ent, attention = self.msa2(c_t, ent, ent, None)
        gent, attention = self.msa3(c_t, gent, gent, None)


        img = self.f_gu(c_t, img)
        enc_src = self.f_gu(c_t, enc_src)
        ent = self.f_gu(c_t, ent)
        gent = self.f_gu(c_t, gent)
        trg = img +  ent + enc_src + gent
        trg = self.dropout(self.v1(trg)) +  self.dropout(self.v2(c_t))
        trg = self.layer_norm(trg)
        _trg = self.GELU(self.positionwise_feedforward(self.GELU(self.positionwise_feedforward1(trg))))
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg

    def f_gu(self, c_t, imgs):
        imgs = imgs + c_t
        imgs = self.v1(imgs) * self.tanh(self.v2(imgs))

        return imgs
