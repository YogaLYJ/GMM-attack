class Config():
    # paramteters clamping range
    param_range = {}
    for i in range(30):
            param_range['GMM_layers.' + str(i) + '.R'] = (-1, 1)

config=Config()

