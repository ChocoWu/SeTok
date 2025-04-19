
if __name__ == "__main__":
    # training setokim
    # from train_setokim import train
    # train(attn_implementation="flash_attention_2")

    # training setok
    from train_setok import train
    train(attn_implementation="flash_attention_2")
