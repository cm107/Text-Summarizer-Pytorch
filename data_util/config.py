# train_data_path = 	"data/chunked/train/train_*"
train_data_path = "data/cnn-dailymail/chunked/train_*"
# valid_data_path = 	"data/chunked/valid/valid_*"
valid_data_path = "data/cnn-dailymail/chunked/val_*"
# test_data_path = 	"data/chunked/test/test_*"
test_data_path = "data/cnn-dailymail/chunked/test_*"
# vocab_path = 		"data/vocab"
vocab_path = "data/cnn-dailymail/vocab"


# Hyperparameters
hidden_dim = 512
# emb_dim = 256
emb_dim = 100 # same as publication
# batch_size = 200
# batch_size = 125 # Needed to lower batch size for MLE+RL
batch_size = 2 # ml+rl: 2, ml: 4; eval: 16 publication calls for 50, but I don't have enough memory
# max_enc_steps = 55		#99% of the articles are within length 55
# max_dec_steps = 15		#99% of the titles are within length 15

# publication calls for 800 encoder steps and 100 decoder steps
max_enc_steps = 1250 # 87.2%
# max_enc_steps = 1500 # 93.8% 1500 is about 2 pages at 12 pt font
# max_enc_steps = 1800 # 97.9% about 2 and a half pages at 12 pt font
# max_enc_steps = 2000 # 99.5% 2000 is about 3 pages at 12 pt font
max_dec_steps = 150 # 99.3% about 2 lines in 12 pt font
# max_dec_steps = 200 # 99.9%
# beam_size = 4
beam_size = 5
min_dec_steps= 3
# vocab_size = 50000
vocab_size = 50000 # orig 50000. Publication calls for 150000, but I don't have enough memory.

lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000


# save_model_path = "data/saved_models"
save_model_path = "data/cnn-dailymail/saved_models"

intra_encoder = True
intra_decoder = True