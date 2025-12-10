import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.UCTransNet import UCTransNet
from utils import *
import cv2

def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(2000,2000))
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()
    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return dice_pred_tmp, iou_tmp

def test_single_dataset(model, test_dataset_path, test_num, vis_path_suffix=""):
    """
    단일 테스트 데이터셋 평가
    
    Args:
        model: 평가할 모델
        test_dataset_path: 테스트 데이터셋 경로
        test_num: 테스트 이미지 개수
        vis_path_suffix: 시각화 결과 저장 경로 접미사 (예: "_TestA")
    
    Returns:
        avg_dice: 평균 Dice Score
        avg_iou: 평균 IoU
    """
    vis_path = "./" + config.task_name + '_visualize_test' + vis_path_suffix + '/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(test_dataset_path, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc=f'Test{vis_path_suffix}', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            
            # Save ground truth
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()
            
            # Predict and save
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                           vis_path+str(i),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    
    avg_dice = dice_pred / test_num
    avg_iou = iou_pred / test_num
    
    return avg_dice, avg_iou

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    model_type = config.model_name
    
    # Task별 설정
    if config.task_name == "GlaS":
        # GlaS는 TestA, TestB 개수 자동 계산
        test_dataset_path = config.test_dataset
        
        # TestA, TestB 개수 확인
        testA_path = test_dataset_path.replace('Test_Folder', 'Test_Folder/TestA')
        testB_path = test_dataset_path.replace('Test_Folder', 'Test_Folder/TestB')
        
        # 현재 test_dataset이 TestA인지 TestB인지 확인
        if 'TestA' in test_dataset_path:
            test_num = len(os.listdir(os.path.join(test_dataset_path, 'img')))
            test_type = "TestA"
        elif 'TestB' in test_dataset_path:
            test_num = len(os.listdir(os.path.join(test_dataset_path, 'img')))
            test_type = "TestB"
        else:
            # Test_Folder 직접 사용 (symbolic link 사용 시)
            test_num = len(os.listdir(os.path.join(test_dataset_path, 'img')))
            test_type = "Test"
        
        model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "MoNuSeg":
        test_num = 14
        test_type = "Test"
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "MLD":
        test_num = len(os.listdir(os.path.join(config.test_dataset, 'img')))
        test_type = "Test"
        model_path = "./MLD/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    else:
        raise ValueError(f"Unknown task: {config.task_name}")

    print(f"Testing {config.task_name} - {test_type} with {test_num} images")
    
    # 모델 로드
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    
    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    elif model_type == 'UCTransNet_pretrain':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    elif model_type == 'UNet':
        from nets.UNet import UNet
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    
    # 단일 테스트 실행
    avg_dice, avg_iou = test_single_dataset(model, config.test_dataset, test_num, 
                                            vis_path_suffix=f"_{test_type}")
    
    print(f"\n{'='*50}")
    print(f"Results for {config.task_name} - {test_type}:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"{'='*50}\n")

