from models.mini_transformer import TransfomerModel
from models.resnet import resnet1d
from models.vgg import VGG1D
from models.mobilenet import MobileNet1D
from models.squeezenet import seresnet1d

''' You only need to change the config_parmas dictionary'''
config_params = dict(
    filepath = "data/outcome_acuity_t14400_v1_14400wd_10hz_0drop.npz",
    target_outcome = "acuity", #brain_status
    model_name = 'mini_transformer',
    fold = 1,
    num_channels = 3,
    downsampling_factor = 8,
    seq_len = 144000,
    num_classes = 2, 
    dataset = 'Acuity',
    d_model = 96,
    num_heads = 16,
    lr = 3e-4,
    eps = 1e-5,
    weight_decay = 1e-5,
    n_epochs = 50,
    bs = 6,
    sampling_mode = "upsampling",
    SEED = 2023,
    )

model_params = dict(
    mini_transformer = TransfomerModel(num_channels=config_params['num_channels'],
                        seq_len=config_params['seq_len'],
                        downsample_factor= int(config_params['downsampling_factor']),
                        dim=int(config_params['d_model']),
                        head_size=int(config_params['num_heads']),
                        num_classes=int(config_params['num_classes'])),

    resnet = resnet1d(int(config_params['num_channels']), int(config_params['num_classes'])),

    mobilenet = MobileNet1D(int(config_params['num_channels']), int(config_params['num_classes'])),                        

    seresnet = seresnet1d(int(config_params['num_channels']), int(config_params['num_classes'])),

    vgg = VGG1D(int(config_params['num_channels']), int(config_params['num_classes'])),
                
                                            )