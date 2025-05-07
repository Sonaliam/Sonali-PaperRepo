class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=302):
        super().__init__()

        # dynamically derive text embedding dimension from RoBERTa
        # self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Load pretrained CLIP model (ViT-B/32)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='openai')
        self.clip_model.visual.eval()
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False

        self.clip_mlp = nn.Sequential(
            nn.Linear(512, hid_dim),  # <== 1024 if that's your feature dim
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim)
        )

        self.pos_embedding = nn.Embedding(hid_dim, hid_dim)

        # project each token embedding to hidden dim
        self.lin = nn.Linear(1024, hid_dim)
        self.lin_i = nn.Linear(1024, hid_dim)
        self.lin_ia = nn.Linear(1024, hid_dim)
        self.lin_e = nn.Linear(1024, hid_dim)
        self.lin_ge = nn.Linear(1024, hid_dim)
        self.lin_f = nn.Linear(512, hid_dim)

        self.msatt_art = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msatt_gent = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm2 = nn.LayerNorm(hid_dim)
        self.msatt_im_art = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msatt_aim_art = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msatt_face_ent = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msatt_img_gent = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.positionwise_feedforward11 = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.positionwise_feedforward12 = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.positionwise_feedforward21 = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.positionwise_feedforward22 = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.GELU = nn.GELU()
        self.hid_dim = hid_dim

    def forward(self, art, art_mask, img, aimg, ent, gent, faces):

        img_pil = ToPILImage()(img.squeeze(0))
        aimg_pil = ToPILImage()(aimg.squeeze(0))

        # Ensure images are in RGB mode (if not already)
        img_pil = img_pil.convert("RGB")
        aimg_pil = aimg_pil.convert("RGB")

        # Preprocess the images using CLIP's preprocessing pipeline
        img = self.clip_preprocess(img_pil).unsqueeze(0)
        aimg = self.clip_preprocess(aimg_pil).unsqueeze(0)

        # Move the tensors to the correct device
        img = img.to(device)
        aimg = aimg.to(device)

        # CLIP feature extraction for main image
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img).float()  # (B, 512)

        # Handle addon image
        if aimg is None or torch.all(aimg == 0):
            presence_flag = 0
            batch_size = img.shape[0]
            aimg_feat = torch.zeros(batch_size, 512).to(img.device)
        else:
            presence_flag = 1
            with torch.no_grad():
                aimg_feat = self.clip_model.encode_image(aimg).float()  # (B, 512)


        img = self.clip_mlp(img_feat).unsqueeze(0)  # (B, hid_dim)
        aimg = self.clip_mlp(aimg_feat).unsqueeze(0)  # (B, hid_dim)
        art = art.mean(dim=1)
        art = self.lin(art)

        batch_size = art.shape[0]
        art_len = art.shape[1]
        pos = torch.arange(0, art_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        art = self.dropout((art * self.scale) + self.pos_embedding(pos))

        art = self.SAF_fea(art, art_mask)
        art = self.CAF_fea(art, art_mask, img, aimg)

        face_presence_flag = 1
        if faces is None or torch.all(faces == 0):
            face_presence_flag = 0
            B = img.size(0)
            faces = torch.zeros((B, 1, 512), device=img.device)
        if faces.dim() == 2:
            faces = faces.unsqueeze(1)

        ent = self.lin_e(ent)
        gent = self.lin_ge(gent)

        ent, gent = self.CEF_fea(img, ent, None, gent, None, faces)

        return art, img, ent, gent

    def CEF_fea(self, img, ent, ent_mask, gent, gent_mask, faces):

        B, N_F, Fd = faces.size()

        F_cat = self.lin_f(
            faces)
        F_tokens = F_cat.view(B, N_F, -1)


        ent_fp_all = []
        for i in range(N_F):
            q = F_tokens[:, i: i + 1, :]  # (B,1,hid_dim)
            attn_out, _ = self.msatt_face_ent(q, ent, ent, ent_mask)  # (B,1,hid_dim)
            x = self.positionwise_feedforward11(attn_out)
            x = self.GELU(x)
            x = self.ff_layer_norm1(self.positionwise_feedforward12(x))
            ent_fp_all.append(x)

        ent_fp = torch.cat(ent_fp_all, dim=1).mean(dim=1, keepdim=True)  # (B,1,hid_dim)
        ent_go, _ = self.msatt_img_gent(img, gent, gent,
                                        gent_mask)  # (B,1,hid_dim) #RuntimeError: Tensors must have same number of dimensions: got 3 and 4
        xg = self.positionwise_feedforward21(ent_go)
        xg = self.GELU(xg)
        ent_go = self.ff_layer_norm2(self.positionwise_feedforward22(xg))  # (B,1,hid_dim)

        return ent_fp, ent_go

    def SAF_fea(self, art, art_mask):
        _art, _ = self.msatt_art(art, art, art, art_mask)
        art = self.self_attn_layer_norm(art + self.dropout(_art))
        return art

    def CAF_fea(self, art, art_mask, img, aimg):
        art_I, _ = self.msatt_im_art(img, art, art, art_mask)

        art_O, _ = self.msatt_aim_art(aimg, art, art, art_mask)

        art = torch.cat((art_I, art_O), 1)
        art = self.positionwise_feedforward(art)
        art = self.GELU(art)
        art = self.ff_layer_norm(self.positionwise_feedforward(art))

        return art

