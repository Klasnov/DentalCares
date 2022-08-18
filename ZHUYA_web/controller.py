from django.shortcuts import render
from django.http import HttpResponse
from . import settings
from .ssd import build_ssd
import torch
import torchvision.models as models
from torch import nn
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from .config import zhuya
from .data_pre import Tooth
import os
import json
from . import save_excel
import zipfile
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data.data_pre import Tooth
import torch.backends.cudnn as cudnn
import time
import tqdm
from torchvision import  transforms as T
from PIL import  Image
from django.http import Http404, HttpResponse,FileResponse,JsonResponse
#import utils
#from utils.response import ReturnCode
save_folder = settings.MEDIA_ROOT + '/detect/'
save_folder2 = settings.MEDIA_ROOT + '/miniapp/'


#小程序
#小程序
def down_image(request):

    if request.method =='GET':
        filename = request.GET.get('filename')
        imagepath = '/opt/ZHUYA_web/static/' + 'media/miniappdetect/'+filename
        image_data =  open(imagepath,"rb").read()
        return HttpResponse(image_data,content_type="image/jpg")

def get_image(request):
    response_data = {}
    if request.method =='GET':

        response_data['is_decayed'] = 'false'
        return HttpResponse(json.dumps(response_data), content_type="application/json")

    if request.method =='POST':

        image = request.FILES['image']
        print(image)
        #open_id=request.POST.get('openid')
        if not os.path.exists(save_folder2+image.name):
            with open(save_folder2+image.name,'wb') as f:
                f.write(image.read())
                f.close()
        detect_path = '/opt/ZHUYA_web/static/' + 'media/' + 'miniapp/' + image.name
        result, score,region = detect_and_cls(detect_path)
        response_data['result'] = result
        response_data['置信度'] = float(score)
        response_data['区域'] = str(region)
        return HttpResponse(json.dumps(response_data), content_type="application/json")

"""单图片上传"""

def deal_single(request):
    image = request.FILES.get('picture')
    if image == None:
        return render(request, 'single_upload.html')

    save_path = settings.MEDIA_ROOT + '/img/' + image.name
    print('save_path', save_path)
    context = {}
    with open(save_path, 'wb') as file:
        for c in image.chunks():
            file.write(c)

    detect_path = '/opt/ZHUYA_web/static/' + 'media/' + 'img/' + image.name
    print('detect_path', detect_path)

    result, score,region = detect_and_cls(detect_path)
    print('score', score)
    print('region',region)
    if result == 'decayed':
        context['result'] = "患有龋齿，需要及时前往口腔科进行治疗。"
    # context['result'] = result
    context['置信度'] = float(score)
    context['区域'] = str(region)
    context['display_path'] = 'media/'+'img/'+image.name
    context['detect_image_path'] = 'media/'+'detect/'+image.name

    return render(request,'single_upload.html',context)
    #解决中文乱码
    # return HttpResponse(json.dumps(context, ensure_ascii=False), content_type="application/json,charset=utf-8")
    # return HttpResponse(json.dumps(context, ensure_ascii=False), content_type="application/json")

def is_ya_single(request):
    image = request.FILES.get('picture')
    if image == None:
        return render(request, 'single_upload.html')
    print("isya")
    save_path = settings.MEDIA_ROOT + '/img/' + image.name
    print('save_path', save_path)
    context = {}
    with open(save_path, 'wb') as file:
        for c in image.chunks():
            file.write(c)

    detect_path = '/opt/ZHUYA_web/static/' + 'media/' + 'img/' + image.name
    print('detect_path', detect_path)

    pred = is_ya(detect_path)
    if pred == 1:
        context['result'] = 'True'
        return HttpResponse(json.dumps(context, ensure_ascii=False), content_type="application/json")

        #return 1
    else:
        context['result'] = 'False'
        return HttpResponse(json.dumps(context, ensure_ascii=False), content_type="application/json")


"""多图片列表上传和多图片上传"""


def deal_multi_list(request):
    image_count = request.POST.get("files_count")
    print(image_count)

    response_data = {}
    image_names = []
    dispaly_path = []
    results = []
    image_sizes = []

    records = []
    dispaly_path = []
    region = []
    for i in range(int(image_count)):

        image_name = request.POST.get(str(i))
        image = request.FILES.get("file" + str(i), None)
        size = request.POST.get('file_size' + str(i))
        save_path = settings.MEDIA_ROOT + '/img/' + image_name

        with open(save_path, 'wb') as file:
            for c in image.chunks():
                file.write(c)

        detect_path = '/opt/ZHUYA_web/static/' + 'media/' + 'img/' + image.name

        result,region = detect_and_cls(detect_path)

        dispaly_path.append("static " + "media/" + "img/" + image_name)
        image_names.append(image_name)
        results.append(result)
        image_sizes.append(size)
        record = []
        record.append(image_name)
        record.append(result)
        records.append(record)

    # print(records)
    # for i in range(len(records)):
    #    for j in range(len(records[i])):
    #       print(records[i][j])
    save_excel_path = save_excel.generate_excel(records)
    response_data['display_path'] = dispaly_path
    response_data['image_names'] = image_names
    response_data['image_sizes'] = image_sizes
    response_data['results'] = results
    response_data['code'] = 0
    response_data['message'] = str(image_count) + '张照片上传成功，请稍等结果'
    response_data['count'] = image_count
    response_data['download_url'] = save_excel_path
    return HttpResponse(json.dumps(response_data), content_type="application/json")

    # return render(request, 'multi_list_upload.html', context)


"""压缩包上传"""


def deal_tar(request):
    zip = request.POST.get('zip');
    zip_name = request.POST.get('zip_name');
    save_path = settings.MEDIA_ROOT + '/img/' + zip_name
    """保存zip"""
    with open(save_path, 'wb') as file:
        for c in zip.chunks():
            file.write(c)

    records = []

    """读取zip"""
    f = zipfile.ZipFile(save_path, 'r')
    for file in f.namelist():
        f.extract(file, settings.MEDIA_ROOT + '/img/' + file.name)
        result,region = detect_and_cls()

    return



# 解决cv2.imread读不了中文名称路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def detect_and_cls(context):
    thresh = 0.60

    # 获取网络
    det_net = settings.det_net
    cls_net = settings.cls_net

    if torch.cuda.is_available():
        det_net = det_net.cuda()
        cls_net = cls_net.cuda()
        cudnn.benchmark = False

    """读取照片"""
    image_path = context
    time_str = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))
    filename = '/opt/ZHUYA_web/static/' + 'media/record/' + image_path.split('/')[-1].split('.')[
        -2] + '_' + time_str + '_test.txt'

    """判断是否有牙"""
    judge = is_ya(image_path)
    print(judge)

    # image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    image = cv_imread(image_path)

    if judge == 0 or image.shape[2] == 4:
        return 'failed', 0,0
    if(1 == 1):
        transform = BaseTransform(det_net.size, (104, 117, 123))
        x = torch.from_numpy(transform(image)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if torch.cuda.is_available():
            x = x.cuda()

        y = det_net(x)
        detections = y.data

        scale = torch.Tensor([image.shape[1], image.shape[0],
                              image.shape[1], image.shape[0]])
        pred_num = 0
        max_score = 0
        max_i = 0
        max_j = 0
        token = 0
        for i in range(detections.size(1)):
            j = 0
            # while detections[0, i, j, 0] >= thresh:
            while detections[0, i, j, 0] >= thresh:
                # 读取图片
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                    f.close()
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                if len(pt) == 0:
                    result = 'NO_Tooth'
                coords = (pt[0], pt[1], pt[2], pt[3])
                print(coords)
                pred_num += 1

                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                f.close()
                print("356", image.shape)  # 这里不应该是[1,3,300,300]
                result, score= crop_cls(image, image_path.split('/')[-1].split('.')[-2], pt, cls_net, j)

                j += 1
                token = 1

            if detections[0, i, j, 0] > 0.1 and detections[0, i, j, 0] > max_score:
                max_score = detections[0, i, j, 0]
                max_i = i
                max_j = j

        if token == 0:

            score = detections[0, max_i, max_j, 0]
            label_name = labelmap[max_i - 1]
            pt = (detections[0, max_i, max_j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            pred_num += 1

            print('>>>>>>>>>>>>>>>>>>', detections[0, max_i, max_j, 0])

            with open(filename, mode='a') as f:
                f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                        str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
            f.close()

            result, score = crop_cls(image, image_path.split('/')[-1].split('.')[-2], pt, cls_net, j)

        # save_path = save_folder + 'Img_test_voc/' + image_path.split('\\')[-1] + '.jpg'
        save_path1 = '/opt/ZHUYA_web/static/media/miniappdetect/' + image_path.split('/')[-1].split('.')[-2] + '.jpg'
        save_path2 = '/opt/ZHUYA_web/static/media/detect/' + image_path.split('/')[-1].split('.')[-2] + '.jpg'
        print(save_path2)
        cv2.imwrite(save_path1, image)
        cv2.imwrite(save_path2, image)
        print('===========================')
        #检测龋齿区域
        img_midpoint = (image.shape[0]/2,image.shape[1]/2)
        #########这里有改动
        region = {(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3]))}
        # region = {(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3]))}
        #region = find_location_region((int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])),img_midpoint)
        #normalregion = "无异常区域"
        if result == 'decayed':
#            return '龋齿', score,region
            return 'decayed', score,region
        elif result == 'healthy':
#            return '正常', score,region
            return 'healthy', score,region


def crop_cls(img, filename, pt, cls_net, j):
    # 裁剪坐标为[y0:y1, x0:x1]
    cropped = img[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]

    print(cropped.shape)
    print("398", filename)

    """预处理当前图片"""
    if cropped.shape[0] > cropped.shape[1]:
        add = int((cropped.shape[0] - cropped.shape[1]) / 2)
        cropped = cv2.copyMakeBorder(cropped, 0, 0, add, add, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        add = int((cropped.shape[1] - cropped.shape[0]) / 2)
        cropped = cv2.copyMakeBorder(cropped, add, add, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    """存取目标六龄牙"""
    # tooth_path = './tooth_crop/' + filename + '_' + str(j) + '.jpg'

    tooth_path = '/opt/ZHUYA_web/static/' + 'media/' + 'crop/' + filename + '_' + str(j) + 'abc.png'
    print(tooth_path)

    cv2.imwrite(tooth_path, cropped)

    """载入dataloader并分类"""
    img_data = Tooth(tooth_path)

    crop_dataloader = DataLoader(img_data, batch_size=1, shuffle=False, num_workers=1)

    for ii, (tooth_data) in enumerate(crop_dataloader):
        print('----------------')
        if torch.cuda.is_available():
            tooth_data = Variable(tooth_data.cuda())
        else:
            tooth_data = Variable(tooth_data)

        socores = cls_net(tooth_data)
        output = torch.nn.functional.softmax(socores)
        output_pro = output[0].cpu().detach().numpy()

        print(tooth_path[0])
        print(output_pro)

        if output_pro[0] > 0.65:
            pre_label = 1
            result = 'decayed'
            pre_pro = round(output_pro[0], 3)
        else:
            pre_label = 0
            result = 'healthy'
            pre_pro = round(output_pro[1], 3)

        print('pre_label', pre_label)
        print('The label is', result)
        print('------------------')

        """标记原始图像"""
        # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255), 3)
        # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
        font = cv2.FONT_HERSHEY_TRIPLEX
        text = result + '-' + str(pre_pro)
        cv2.putText(img, text, (int(pt[0]) + 5, int(pt[1] - 10)), font, 1, (255, 0, 0), 1)


        return result, pre_pro

#判断图片是不是牙
def is_ya(image_path):

    model = settings.model
    context = {}
    if torch.cuda.is_available():
        model = model.cuda()

    transfrom = T.Compose(
        [
            T.CenterCrop(224),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = transfrom(image)
    image = image.unsqueeze(0)
#    image = image.cuda()

    score = model(image)
    score = F.softmax(score, dim=1)
    pred = torch.argmax(score, dim=1)

    return pred


        #return 0

#比较蛀牙区域中间点与图像中间点坐标
def find_location_region(leftup,rightbottom,img_midpoint):
	    region = ""
	    tooth_midpoint = ((leftup[0]+rightbottom[0])/2,(leftup[1]+rightbottom[1])/2)
	    if(tooth_midpoint[0]<=img_midpoint[0] and tooth_midpoint[1]<=img_midpoint[1]):
	        region = "左上侧"
	    elif(tooth_midpoint[0]<=img_midpoint[0] and tooth_midpoint[1]>=img_midpoint[1]):
	        region = "左下侧"
	    elif(tooth_midpoint[0]>=img_midpoint[0] and tooth_midpoint[1]<=img_midpoint[1]):
	        region = "右上侧"
	    elif(tooth_midpoint[0]>=img_midpoint[0] and tooth_midpoint[1]>=img_midpoint[1]):
	        region = "右下侧"
	    else:
	        region = "中部"

	    return region
