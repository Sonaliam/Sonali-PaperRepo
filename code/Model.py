class NICModel1(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)


        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()


        trg_mask = trg_pad_mask & trg_sub_mask


        return trg_mask

    def forward(self, src, trg, img, aimg, ent, gent, face):  #reference

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src, img, ent, gent  = self.encoder(src, src_mask, img, aimg, ent, gent, face)
        output = self.decoder(trg, enc_src, src, trg_mask, src_mask, img, ent, gent)


        return output, src_mask

