import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
epochs = 2000
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False

# ============= TASK 선택 =============
task_name = 'MoNuSeg'  # 의료 영상 - 세포핵 분할
# task_name = 'MLD'      # Massachusetts Roads Dataset
# task_name = 'GlaS'       # Gland Segmentation Dataset
# =====================================

learning_rate = 1e-3
batch_size = 4

# model_name = 'UCTransNet'
model_name = 'UNet'

# Task별 설정
if task_name == 'MoNuSeg':
    n_channels = 3
    n_labels = 1
    img_size = 224
elif task_name == 'MLD':
    n_channels = 3      # RGB 이미지
    n_labels = 1        # 이진 분할 (도로 vs 배경)
    img_size = 224      # 원본 1500x1500을 224로 리사이즈
elif task_name == 'GlaS':
    n_channels = 3      # RGB 병리 이미지 (H&E 염색)
    n_labels = 1        # 이진 분할 (gland vs background)
    img_size = 224      # 원본 ~775x522를 224로 리사이즈
else:
    raise ValueError(f"Unknown task: {task_name}")

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/TestB'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'

##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

# used in testing phase, copy the session name in training phase
test_session = "Test_session_12.10_08h35"

