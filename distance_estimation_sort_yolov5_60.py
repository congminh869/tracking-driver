import sys
sys.path.insert(0, './yolov5')
sys.path.insert(0, '/home/minhssd/Documents/Doan/yolov5_sort/classy-sort-yolov5/sort')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

#yolov5
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
from utils.datasets import letterbox
# from yolov5.utils.plots import Annotator
#SORT
import skimage
from sort import *

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cat = int(categories[i]) if categories is not None else 0
        
        id = int(identities[i]) if identities is not None else 0
        
        color = compute_color_for_labels(id)
        
        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, 1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img

def xyxy2xywh2(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[0] + x[2]) / 2  # x center
    y[:, 1] = (x[1] + x[3]) / 2  # y center
    y[:, 2] = x[2] - x[0]  # width
    y[:, 3] = x[3] - x[1]  # height
    return y
def FocalLength(measured_distance, real_width, width_in_rf_image):
    # Function Discrption (Doc String)
    """
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
    :param1 Measure_Distancue(int): It is distance measured from object to the Camera while Capturing Reference image

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
    :retrun Focal_Length(Float):
    """
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# distance estimation function

def config_obj(name, face_width_in_frame):
    if name == 'face':
        Known_distance = 70 #real distance
        Known_width = 15#real_face_width
        ref_image_face_width = 123#pixels
        Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
        distance = (Known_width * Focal_length_found) / face_width_in_frame
        distance = round(distance, 2)
        return distance
    elif names=='car':
        Known_distance = 70 #real distance
        Known_width = 15#real_face_width
        ref_image_face_width = 123#pixels
        Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
        distance = (Known_width * Focal_length_found) / face_width_in_frame
        distance = round(distance, 2)
        return distance
    elif names=='bus':
        Known_distance = 70 #real distance
        Known_width = 15#real_face_width
        ref_image_face_width = 123#pixels
        Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
        distance = (Known_width * Focal_length_found) / face_width_in_frame
        distance = round(distance, 2)
        return distance
    elif names=='truck': #motorcycle
        Known_distance = 70 #real distance
        Known_width = 15#real_face_width
        ref_image_face_width = 123#pixels
        Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
        distance = (Known_width * Focal_length_found) / face_width_in_frame
        distance = round(distance, 2)
        return distance
    elif names=='motorcycle':
        Known_distance = 70 #real distance
        Known_width = 15#real_face_width
        ref_image_face_width = 123#pixels
        Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
        distance = (Known_width * Focal_length_found) / face_width_in_frame
        distance = round(distance, 2)
        return distance
    return 0

def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
    global count_traffic
    global count_id
    global traffic 
    global id_pixel1_pixel2
    global distance_ests
    high, weight = img_detect.shape[:2] #(chieu cao, chieu rong, chieu sau)
    # print('********************')
    # print(high, weight) #444 640
    # print('********************')
    #####################################
    classify = False
    agnostic_nms = False
    augment = False
    # Set Dataloader
    #vid_path, vid_writer = None, None
    # Get names and colors
    

    count = 0
    t = time.time()
    #processing images
    '''
    Tiền xử lí ảnh, numpy() => muon dua anh vao 1 model AI thi minh phai convert no ve kieu cua model, tensor()
    '''
    im0 = letterbox(img_detect, img_size)[0]
    im0 = im0[:, :, ::-1].transpose(2, 0, 1)
    im0 = np.ascontiguousarray(im0)
    im0 = torch.from_numpy(im0).to(device)
    im0 = im0.half() if half else im0.float()
    im0 /= 255.0  # 0 - 255 to 0.0 - 1.0 RGB chuan hoa cac diem anh ve 0->1
    if im0.ndimension() == 3:
        im0 = im0.unsqueeze(0)
    # Inference
    t1 = time.time()

    #bat dau model pre
    pred = model(im0, augment= augment)[0]
    #output
    # print('pred : ')
    # print(pred)
    
    # print('time detect : ', t2 - t1)
    # Apply NMS
    classes = [2,3,5,7]#None
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
    # Apply Classifier 
    if classify:
        pred = apply_classifier(pred, model, im0, img_ocr)
    gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
    points = []

    if len(pred[0]):
        check = True
        pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
        xywhs = xyxy2xywh(pred[0][:, 0:4])
        confs = pred[0][:, 4]
        clss = pred[0][:, 5]
        dets_to_sort = np.empty((0,6))

        for x1,y1,x2,y2,conf,detclass in pred[0].cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

        tracked_dets = sort_tracker.update(dets_to_sort)
        t2 = time.time()
        denta_t = t2-t1
        if len(tracked_dets)>0:
            for box in tracked_dets:
                cls = int(box[4])
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                c1 = (int(x1), int(y1))
                c2 = (int(x2), int(y2))
                label = names[cls]
                face_width = int(x2-x1)
                id = int(box[8])
                # print(label + " " + str(id))
                if label in traffic:
                    Distance = config_obj(label, face_width)
                    if id not in id_pixel1_pixel2:
                        id_pixel1_pixel2[id] = []
                        distance_ests[id] = []
                        id_pixel1_pixel2[id].append(face_width)
                        distance_ests[id].append(Distance)
                    else:
                        id_pixel1_pixel2[id].append(face_width)
                        distance_ests[id].append(Distance)

                    if len(id_pixel1_pixel2[id])>=3:
                        del id_pixel1_pixel2[id][0]
                        del distance_ests[id][0]
                    # print('len(id_pixel1_pixel2[id])>=2 : ', len(id_pixel1_pixel2[id]))
                    # print('len(id_pixel1_pixel2[id])>=2 : ', id_pixel1_pixel2[id])
                    if len(id_pixel1_pixel2[id])>=2:
                        if abs(id_pixel1_pixel2[id][0]-id_pixel1_pixel2[id][1])<=3:
                            id_pixel1_pixel2[id][0]=id_pixel1_pixel2[id][1]

                        if id_pixel1_pixel2[id][0]==id_pixel1_pixel2[id][1]:
                            time_collision = 0
                            cv2.putText(img_detect, str(abs(id_pixel1_pixel2[id][0]-id_pixel1_pixel2[id][1]))+' '+ str(id_pixel1_pixel2[id]) +' '+ str(round(time_collision,3)) , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,51), 1, cv2.LINE_AA)
                            cv2.rectangle(img_detect, c1, c2, (255, 255, 51), 2)
                        else:
                            time_collision = denta_t/(id_pixel1_pixel2[id][1]/id_pixel1_pixel2[id][0]-1)
                            if time_collision<0.1:
                                cv2.putText(img_detect, str(abs(id_pixel1_pixel2[id][0]-id_pixel1_pixel2[id][1]))+' '+ str(id_pixel1_pixel2[id]) +' '+ str(round(time_collision,3)) , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                                cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 2)
                            elif time_collision>18:
                                continue
                            else:
                                cv2.putText(img_detect, str(abs(id_pixel1_pixel2[id][0]-id_pixel1_pixel2[id][1]))+' '+ str(id_pixel1_pixel2[id]) +' '+ str(round(time_collision,3)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,51), 1, cv2.LINE_AA)
                                cv2.rectangle(img_detect, c1, c2, (255, 255, 51), 2)
                        print(str(time_collision) + ' ' +str(distance_ests[id]))
                        
    print(id_pixel1_pixel2)
    print(distance_ests)
    return img_detect

if __name__ == '__main__':
    global id_pixel1_pixel2
    global distance_ests
    distance_ests = {}
    id_pixel1_pixel2 = {}
    traffic = ['car', 'motorcycle', 'bus', 'truck']
    use_gpu = torch.cuda.is_available()
    print('use_gpu : ',use_gpu)
    img_size = 320
    conf_thres = 0.4
    iou_thres = 0.45
    device = ''
    update = True
    # Load model yolo
    print('=================Loading models yolov5=================')
    t1 = time.time()
  #check co GPU ko 
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model nhan dien container
    t1 = time.time()
  #path file trong so 
    weights = './weights/yolov5s.pt' 
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    print('names : ', names)
    if half:
        model.half()
    t2 = time.time()
    print('time load model yolo : ', t2-t1)

    #path file image test
    # frame = cv2.imread('test.png')
    #dau ra la anh da detect
    # frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 320)


    # sort
    sort_max_age = 30
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) # {plug into parser}
    
  # cv2.imwrite('result.png', frame)
    cap = cv2.VideoCapture('/home/minhssd/Pictures/11.mp4')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)

    output = cv2.VideoWriter('./output/output_videotest1.mkv', cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_size)
    count_frame = 0
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if count_frame%20==0:
            frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.4, img_size = 640)
            output.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow("frame", frame)
            time.sleep(1)
        count_frame+=1

    cap.release()
    result.release()
    cv2.destroyAllWindows()
# a= 400
# for i in range(2,800):
#   denta_t = 0.11
#   time_collision = denta_t/((a+i)/a - 1) 
#   print(str(a+i)      +' : '+ str(time_collision))