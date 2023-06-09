dataset_parameters = {
    'mfmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'fmnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist_cifar': {
        'in_shape': [10, 3, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist_cifar',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'taxibj': {
        'in_shape': [4, 2, 32, 32],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'human': {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['kth20', 'kth'], {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 20,
        'total_length': 30,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    }),
    'kth40': {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 40,
        'total_length': 50,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'kitticaltech': {
        'in_shape': [10, 3, 128, 160],
        'pre_seq_length': 10,
        'aft_seq_length': 1,
        'total_length': 11,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'mv_weather_4_4_s3_d1_40625': {  # multi-variant weather bench, 4->4, step=3, deg1.40625
        'in_shape': [4, 38, 128, 256],
        'out_shape': [4, 5, 128, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'in_var_list': ['z', 'q', 't', 'u', 'v', 't2m', 'u10', 'v10'],
        'out_var_list': ['z', 'q', 't', 'u', 'v'],
        'in_level_list': [100, 300, 500, 700, 850, 925, 1000],
        'out_level_list': [500],
        'in_time_len': 4, 'out_time_len': 4, 'out_time_res': 2,
        'shift_step': 3,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'mv_weather_4_4_s6_d1_40625': {  # multi-variant weather bench, 4->4, step=6, deg1.40625
        'in_shape': [4, 38, 128, 256],
        'out_shape': [4, 5, 128, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'in_var_list': ['z', 'q', 't', 'u', 'v', 't2m', 'u10', 'v10'],
        'out_var_list': ['z', 'q', 't', 'u', 'v'],
        'in_level_list': [100, 300, 500, 700, 850, 925, 1000],
        'out_level_list': [500],
        'in_time_len': 4, 'out_time_len': 4, 'out_time_res': 2,
        'shift_step': 6,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'mv_weather_4_4_s12_d1_40625': {  # multi-variant weather bench, 4->4, step=12, deg1.40625
        'in_shape': [4, 38, 128, 256],
        'out_shape': [4, 5, 128, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'in_var_list': ['z', 'q', 't', 'u', 'v', 't2m', 'u10', 'v10'],
        'out_var_list': ['z', 'q', 't', 'u', 'v'],
        'in_level_list': [100, 300, 500, 700, 850, 925, 1000],
        'out_level_list': [500],
        'in_time_len': 4, 'out_time_len': 4, 'out_time_res': 2,
        'shift_step': 12,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'mv_weather_4_4_s24_d1_40625': {  # multi-variant weather bench, 4->4, step=24, deg1.40625
        'in_shape': [4, 38, 128, 256],
        'out_shape': [4, 5, 128, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'in_var_list': ['z', 'q', 't', 'u', 'v', 't2m', 'u10', 'v10'],
        'out_var_list': ['z', 'q', 't', 'u', 'v'],
        'in_level_list': [100, 300, 500, 700, 850, 925, 1000],
        'out_level_list': [500],
        'in_time_len': 4, 'out_time_len': 4, 'out_time_res': 2,
        'shift_step': 24,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_5_625': {  # multi-variable weather bench
        'in_shape': [2, 18, 32, 64],
        'out_shape': [24, 8, 32, 64],
        'pre_seq_length': 2,
        'aft_seq_length': 24,
        'total_length': 26,
        'step': 6,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_1_40625': {  # multi-variable weather bench
        'in_shape': [2, 18, 128, 256],
        'out_shape': [24, 8, 128, 256],
        'pre_seq_length': 2,
        'aft_seq_length': 24,
        'total_length': 26,
        'step': 6,
        'metrics': ['mse', 'rmse', 'mae'],
    },
    **dict.fromkeys(['weather', 'weather_t2m_5_625'], {  # 2m_temperature
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    }),
    'weather_mv_4_28_s6_5_625': {  # multi-variant weather bench, 4->28 (7 days)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 28,
        'total_length': 32,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(28)],
        'step': 6,
        'level': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_4_4_s6_5_625': {  # multi-variant weather bench, 4->4 (1 day)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(4)],
        'step': 6,
        'level': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_5_625': {  # relative_humidity
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_5_625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_5_625': {  # total_cloud_cover
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_t2m_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_1_40625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_1_40625': {  # total_cloud_cover
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
}