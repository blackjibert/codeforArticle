lr= 0.70
N_SERVER = 4  #数据服务器数量
STALENESS_THRESHOLD = 100    # 异步程度
STDEV = 1000  # 数据不平衡程度

模型使用的resnet18
无噪声
// 噪声大小为gussNoise = gaussian_mech_RDP_vec(0.01, 0.1, 1, dict_params[parms].shape)
