from models.mini_transformer import TransfomerModel
from models.resnet import resnet1d
from models.vgg import VGG1D
from models.mobilenet import MobileNet1D
from models.squeezenet import seresnet1d

''' You only need to change the config_parmas dictionary'''
config_params = dict(
    filepath = "data/outcomedelirium_t10800_delirium_10800wd_10hz_0drop.npz",
    target_outcome = "brain_status",
    model_name = 'mini_transformer',
    fold = 3,
    num_channels = 3,
    seq_len = 108000//3,
    num_classes = 3, 
    dataset = 'Delirium',
    d_model = 128,
    num_heads = 16,
    lr = 1e-3,
    eps = 1e-5,
    weight_decay = 1e-5,
    n_epochs = 50,
    bs = 4,
    sampling_mode = "upsampling",
    SEED = 2023,
    )

model_params = dict(
    mini_transformer = TransfomerModel(num_channels=config_params['num_channels'],
                        seq_len=config_params['seq_len'],dim=int(config_params['d_model']),
                        head_size=int(config_params['num_heads']),
                        num_classes=int(config_params['num_classes'])),

    resnet = resnet1d(int(config_params['num_channels']), int(config_params['num_classes'])),

    mobilenet = MobileNet1D(int(config_params['num_channels']), int(config_params['num_classes'])),                        

    seresnet = seresnet1d(int(config_params['num_channels']), int(config_params['num_classes'])),

    vgg = VGG1D(int(config_params['num_channels']), int(config_params['num_classes'])),
                
                                            )