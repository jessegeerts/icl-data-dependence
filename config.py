from utils import dotdict as dd


config = dd(dict(
    model=dd(dict(
        h_dim=128,                      # hidden dimensionality of the transformer model
        n_heads=1,
        n_blocks=2,
        include_mlp=[False, True],
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=True,
        widening_factor=1,              # how much wider is the MLP hidden dim
        max_T=32,                       # max sequence length for the model
        out_dim=None,                    # note this is set later (dependent on N labels in data)
        drop_p=0.0
    )),
    data=dd(dict(
        S=10000,
        K=2**10,                 # number of classes
        L=32,                   # number of labels
        D=63,                   # dimension of inputs
        alpha=0.,               # zipf exponent
        eps=0.75,                # within-class variance (higher => more ICL)
    )),
    seq=dd(dict(
        N=8,                   # sequence length will be 2N + 1
        B=4,
        pB=1.,
        pC=1.,
    )),
    train=dd(dict(
        batch_size=128,
        learning_rate=.01,          # learning rate for SGD
        learning_rate_adam=1e-3,    # learning rate for Adam
        w_decay=1e-7,               # L2 regularisation parameter. note the torch implementation is a bit different from reddy jax code (it's multiplied by LR, so divide by LR to get desired w_decay param )
        niters=150000*2,
        optim='adam'                # 'adam' or 'sgd'
    )),
    log_to_wandb=True,
    logging_interval=500,  # iterations
    save_weights=True
))
