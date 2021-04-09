import experiment_buddy

batch_size = 32
lr = 0.01
momentum = 0.9

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)