method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 128
hid_T = 512
N_T = 8
N_S = 2

# training
# lr = 1e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5