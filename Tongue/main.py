import os
import time
import json
import os
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from src import deeplabv3_resnet50
import matplotlib.pyplot as plt
from model import create_regnet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def merge_imgandmask(images_path , masks_path , masked_path):
  for img_item in os.listdir(images_path):
    img_path = os.path.join(images_path,img_item)
    img = cv2.imread(img_path)             #读入img的三通道原图

    b,g,r = cv2.split(img)
    mask_path = os.path.join(masks_path, img_item[:-4] + '.png')  # mask是.png格式的，image是.jpg格式的
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取


    masked_b = cv2.add(b, np.zeros(np.shape(b), dtype=np.uint8), mask=mask)  # 将image的相素值和mask像素值相加得到结果
    masked_g = cv2.add(g, np.zeros(np.shape(b), dtype=np.uint8), mask=mask)  # 将image的相素值和mask像素值相加得到结果
    masked_r = cv2.add(r, np.zeros(np.shape(b), dtype=np.uint8), mask=mask)  # 将image的相素值和mask像素值相加得到结果

    masked = cv2.merge([masked_b,masked_g,masked_r])
    cv2.imwrite(os.path.join(masked_path, img_item), masked)


def resize(path):
    for maindir, subdir,file_name_list in os.walk(path):
       print(file_name_list)
       for file_name in file_name_list:
        image=os.path.join(maindir,file_name) #获取每张图片的路径
        file=Image.open(image)
        out=file.resize((780,520),Image.ANTIALIAS)  #以高质量修改图片尺寸为（400，48）
        out.save(image)



def main():
    aux = False  # inference time not need aux_classifier
    classes = 1
    weights_path = "./save_weights/Segmentation.pth"

    dir_origin_path = "tongue_data/img/"
    dir_save_path   = "tongue_data/img_out/"
    masked_save_path = "tongue_data/masked/"

    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = deeplabv3_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            img_path = os.path.join(dir_origin_path, img_name)
            original_img = Image.open(img_path)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            if not os.path.exists(masked_save_path):
                os.makedirs(masked_save_path)
        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(520),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            mask.save(os.path.join(dir_save_path, img_name[:-4]+'.png'))

    resize(dir_origin_path)
    merge_imgandmask(dir_origin_path,dir_save_path,masked_save_path)




#分类任务
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform2 = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # dir_save_path = "./tongue_data/test_out"
    img_names = os.listdir(masked_save_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            img_path = os.path.join(masked_save_path, img_name)
            # original_img = Image.open(img_path)
            # if not os.path.exists(dir_save_path):
            #     os.makedirs(dir_save_path)
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform2(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_regnet(model_name="RegNetY_400MF", num_classes=3).to(device)
        # load model weights
        model_weight_path = "./save_weights/Classcification.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        # cv2.imwrite(dir_save_path,img)
        plt.show()


if __name__ == '__main__':
    main()
