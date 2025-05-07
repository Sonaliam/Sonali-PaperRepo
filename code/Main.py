SPECIALS2IDX = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
nb_tokens  = len(data_loader_train.dataset.vocab)
decoder_lr = 0.000001
fine_tune_encoder = False
image_size = 256
hid_dim = 1024
pf_dim = 256
crop_size = 224
best_bleu4 = 1
epochs_since_improvement = 1
model_path = "checkpoint/"
clip = 1

data_name = ''
checkpoint = None
start_epoch = 0

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def prepare_input1(resolution):
    src = torch.FloatTensor(1, 512, 1024)
    trg = torch.randint(low=0, high=10, size=(1, 13)) #torch.LongTensor(1, 13)     # trg = (0, 2, 2) #'paul krugman'
    img = torch.FloatTensor(1, 6, 1024)
    aimg= torch.FloatTensor(1, 6, 1024)
    ent= torch.FloatTensor(1, 20, 1024)
    gent= torch.FloatTensor(1, 20, 1024)
    face= torch.FloatTensor(1, 4, 512)

    if torch.cuda.is_available():
        # net.cuda(device=args.device)
        src = src.to(device)
        trg = trg.to(device)
        img = img.to(device)
        aimg = aimg.to(device)
        ent = ent.to(device)
        gent = gent.to(device)
        face = face.to(device)
    return dict(src=src, trg=trg, img=img, aimg=aimg, ent=ent, gent=gent, face=face)

def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, train_logger, dev_logger


    if checkpoint is None:
        enc = Encoder(len(vocab), hid_dim, 1, 1, pf_dim, 0.1)
        dec = Decoder(len(vocab), hid_dim, 6, 1, pf_dim, 0.1) #output_dim,  hid_dim,  n_layers,   n_heads,  pf_dim,  dropout,  max_length=38):
        model = NICModel1(enc, dec, 0, 0)
        optimizer = optim.Adam(model.parameters(), lr=decoder_lr)

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(initialize_weights)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['decoder']
        optimizer = optim.Adam(model.parameters(), lr=decoder_lr)

        model = model.to(device)

    flops, params = get_model_complexity_info(model, input_res=((1, 512, 1024), (1, 13), (1, 6, 1024), (1, 6, 1024),(1, 10, 300),(1, 10, 300),(1, 4, 512)),input_constructor=prepare_input1, as_strings=True, print_per_layer_stat=True)

    print('FLOPs:', flops)

    print('Parameters:', params)


    train_log_dir = os.path.join(model_path, 'train')
    dev_log_dir = os.path.join(model_path, 'dev')
    train_logger = Logger(train_log_dir)
    dev_logger = Logger(dev_log_dir)

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIALS2IDX['<pad>'])

    # Image preprocessing, normalization for the pretrained resnet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    for epoch in range(start_epoch, epochs):
        train(model = model,
              data_loader_train=data_loader_train,
              criterion=criterion,
              optimizer = optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        if epoch > 0:
            recent_bleu4 = validate(model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    vocab=vocab,
                                    epoch=epoch,
                                    logger=dev_logger,
                                    logging=True)

            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            print('best_bleu4:', best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpoch since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

        if epoch <= 4:
            recent_bleu4 = 0
            is_best = 1

        save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer, recent_bleu4, is_best)