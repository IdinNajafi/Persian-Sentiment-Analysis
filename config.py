from keras import regularizers

max_features = 200000
max_senten_len = 32
max_senten_num = 8
embed_size = 150
max_vocab_size = 5000
VALIDATION_SPLIT = 0.2
hidden_dim = 100
dropout = 0.1
REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)

batch_size = 16
epochs = 1
LEARNING_RATE = 2e-5
seed=2