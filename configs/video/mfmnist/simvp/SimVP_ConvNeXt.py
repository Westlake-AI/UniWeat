method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'convnext'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-2
batch_size = 16
drop_path = 0
sched = 'onecycle'