import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import torch
import time
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
from utils.datasets import letterbox


import threading
from threading import Lock
#https://github.com/lincolnhard/head-pose-estimation
#https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()

    def __init__(self, rtsp_link):
        capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(capture,), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()


    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None

def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
    global COUNTER
    global MOUTH_COUNTER
    imgsz = img_size
    high, weight = img_detect.shape[:2]
    check = False
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
    Tiền xử lí ảnh
    '''
    im0 = letterbox(img_detect, img_size)[0]
    im0 = im0[:, :, ::-1].transpose(2, 0, 1)
    im0 = np.ascontiguousarray(im0)
    im0 = torch.from_numpy(im0).to(device)
    im0 = im0.half() if half else im0.float()
    im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im0.ndimension() == 3:
        im0 = im0.unsqueeze(0)
    # Inference
    t1 = time.time()
    pred = model(im0, augment= augment)[0]
    t2 = time.time()
    # print('---------------------------------time detect : ', t2 - t1)
    # Apply NMS
    classes = None#[0, 67]
    # classes = [0, 67]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, im0, img_ocr)
    gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
    points = []
    img_crops = []
    x_y_center = []
    cs=[]
    if len(pred[0]):
        check = True
        pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
        for c in pred[0][:, -1].unique():
            n = (pred[0][:, -1] == c).sum()  # detections per class
        for box in pred[0]:
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            x1, y1 = c1
            x2, y2 = c2
            x_center = int((x1+x2)/2)
            y_center = int((y1+y2)/2)
            acc = round(float(box[4])*100,2)
            cls = int(box[5])
            conf = box[4].item()
            label = names[cls]#
            # print(label)
            
            img_crop = img_detect[y1:y2, x1:x2]
            # cv2.rectangle(img_detect, c1, c2, (255,255,51), 2)
            # cv2.putText(img_detect,label + ' ' + str( round(conf, 3)) + ' ' + str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,51), 2, cv2.LINE_AA)

            if label == 'phone':
                cv2.putText(img_detect,"Don't use the phone", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            if label == 'smoke':
                cv2.putText(img_detect,'No smoking', (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            if label=='face':
                cv2.rectangle(img_detect, c1, c2, (255,255,51), 2)
                cv2.putText(img_detect,label + ' ' + str( round(conf, 3)) + ' ' + str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,51), 2, cv2.LINE_AA)
                print('--------------------------------------------------------------------------------------')
                dlibRect = dlib.rectangle(x1, y1, x2, y2)
                shape = predictor(img_detect, dlibRect)
                # print()
                print('len(shape) predictor: ', shape)
                shape = face_utils.shape_to_np(shape)

                print('len(shape) : ', shape)

                leftEye = shape[lStart:lEnd]
                print('len(leftEye) : ', len(leftEye))
                print('len(leftEye) : ', leftEye)
                rightEye = shape[rStart:rEnd]
                print('len(rightEye) : ', len(rightEye))
                print('len(rightEye) : ', rightEye)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                mouth = shape[mouth_start:mouth_end]
                mouthar = mouth_aspect_ratio(mouth)
                # print('mouthar : ', mouthar)
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)

                # cv2.drawContours(img_detect, [leftEyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(img_detect, [rightEyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(img_detect, [mouthHull], -1, (0, 255, 255), 1)
                
                reprojectdst, euler_angle = get_head_pose(shape)

                # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], #Array of corresponding image points
                #                         shape[39], shape[42], shape[45], shape[31], shape[35], #2xN ( or Nx2 ) 1-channel or 1xN ( or Nx1 )
                #                         shape[50], shape[52], shape[62], shape[66]])            #where N is the number of points.
                list_test = [shape[36], #Array of corresponding image points
                                        shape[45], shape[33], shape[48], shape[54],  shape[8]]
                index_lenmark = 0
                # for (x, y) in shape:#list_test:#
                #     # cv2.circle(img_detect, (x, y), 1, (0, 0, 255), -1)
                #     cv2.putText(img_detect, str(index_lenmark), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.3, (0, 0, 255), thickness=1)
                #     index_lenmark += 1
                # print(reprojectdst)
                # for start, end in line_pairs:
                #     start_point = (abs(int(reprojectdst[start][0])), abs(int(reprojectdst[start][1])))
                #     end_point = (abs(int(reprojectdst[end][0])) , abs(int(reprojectdst[end][1])))
                #     # print('reprojectdst[start] : ', reprojectdst[start][0])
                #     # print('reprojectdst[end] : ', reprojectdst[end])
                #     # if start_point[0] < 0 or start_point[0] > int(width/2) or start_point[0] > int(height/2) or end_point[0] < 0 or end_point[0] > int(width/2) or end_point[0] > int(height/2) or start_point[1] < 0 or start_point[1] > int(width/2) or start_point[1] > int(height/2) or end_point[1] < 0 or end_point[1] > int(width/2) or end_point[1] > int(height/2):
                #     if start_point[0] < 0 or end_point[0] < 0 or start_point[1] < 0 or end_point[1] < 0 or start_point[1] >1000 or start_point[0] >1000 or end_point[1] > 1000 or end_point[0] > 1000:
                #         continue
                #     cv2.line(img_detect, start_point, end_point, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(img_detect, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(img_detect, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(img_detect, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                cv2.putText(img_detect, "mouth : " + "{:7.2f}".format(mouthar), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                cv2.putText(img_detect, "ear : " + "{:7.2f}".format(ear), (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                check_head_pose = 33

                if  abs(int(euler_angle[0, 0])) >= check_head_pose or abs(int(euler_angle[1, 0])) >= check_head_pose:
                    cv2.putText(img_detect, "focus work", (20, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if mouthar > MOUTH_AR_THRESH:
                    MOUTH_COUNTER += 1
                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        # draw an alarm on the frame
                        cv2.putText(img_detect, "you are yawning", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    MOUTH_COUNTER = 0

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # draw an alarm on the frame
                        cv2.putText(img_detect, "Don't Sleep", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0

    return img_detect

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def mouth_aspect_ratio(mouth): 
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], #Array of corresponding image points
                            shape[39], shape[42], shape[45], shape[31], shape[35], #2xN ( or Nx2 ) 1-channel or 1xN ( or Nx1 )
                            shape[48], shape[54], shape[57], shape[8]])            #where N is the number of points.

    # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], #Array of corresponding image points
    #                         shape[39], shape[42], shape[45], shape[31], shape[35], #2xN ( or Nx2 ) 1-channel or 1xN ( or Nx1 )
    #                         shape[50], shape[52], shape[62], shape[66]])            #where N is the number of points.

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # print('rotation_vec : ', rotation_vec.shape) #(3, 1)
    # print('translation_vec : ', translation_vec.shape) #(3, 1)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # print('reprojectdst : ', len(reprojectdst)) #8

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    # print('rotation_mat : ', rotation_mat.shape) #(3,3)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec)) # used to combine images of same height horizontally.
    # print('pose_mat : ', pose_mat.shape) #(3,4)
    # print('pose_mat : ', pose_mat) #[[ 9.99573509e-01 -2.09843238e-02 -2.03090810e-02  9.99117608e+00]
                                     # [ 2.00648068e-02  9.98809325e-01 -4.44672491e-02 -7.87951457e+00]
                                     # [ 2.12180147e-02  4.40407864e-02  9.98804388e-01 -1.19174865e+02]]
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    # print('euler_angle : ', euler_angle.shape) #(3,1)

    return reprojectdst, euler_angle


if __name__ == '__main__':
    # return
    # cap = cv2.VideoCapture(0)#

    #load model yolo 
    face_landmark_path = '/home/minhssd/Downloads/shape_predictor_68_face_landmarks_GTX.dat'

    K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
         0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
         0.0, 0.0, 1.0]
    D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32) #Input camera matrix A fx, fy can be image width , cx, cy can be image center
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32) # Input vector of distortion coefficients  of 4, 5, 8 or 12 elements

    object_pts = np.float32([[6.825897, 6.760612, 4.402142], #Array of object points in the world coordinate space
                             [1.330353, 7.122144, 6.903745], #Nx3 ( or 3xN ) single channel matrix, or Nx1 ( or 1xN ) 3 channel matrix.
                             [-1.330353, 7.122144, 6.903745],
                             [-6.825897, 6.760612, 4.402142],
                             [5.311432, 5.485328, 3.987654],
                             [1.789930, 5.393625, 4.413414],
                             [-1.789930, 5.393625, 4.413414],
                             [-5.311432, 5.485328, 3.987654],
                             [2.005628, 1.409845, 6.165652],
                             [-2.005628, 1.409845, 6.165652],
                             [2.774015, -2.080775, 5.048531],
                             [-2.774015, -2.080775, 5.048531],
                             [0.000000, -3.116408, 6.097667],
                             [0.000000, -7.415691, 4.070434]])

    reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                               [10.0, 10.0, -10.0],
                               [10.0, -10.0, -10.0],
                               [10.0, -10.0, 10.0],
                               [-10.0, 10.0, 10.0],
                               [-10.0, 10.0, -10.0],
                               [-10.0, -10.0, -10.0],
                               [-10.0, -10.0, 10.0]])

    line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4],
                  [0, 4], [1, 5], [2, 6], [3, 7]]

    EYE_AR_THRESH = 0.25 #  If the eye aspect ratio falls below this threshold, we’ll start counting the number of frames the person has closed their eyes for.
    EYE_AR_CONSEC_FRAMES = 10 # If the number of frames the person has closed their eyes in exceeds EYE_AR_CONSEC_FRAMES (Line 49), we’ll sound an alarm.
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0

    MOUTH_AR_THRESH = 0.65 #  If the eye aspect ratio falls below this threshold, we’ll start counting the number of frames the person has closed their eyes for.
    MOUTH_AR_CONSEC_FRAMES = 5 # If the number of frames the person has closed their eyes in exceeds EYE_AR_CONSEC_FRAMES (Line 49), we’ll sound an alarm.
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    MOUTH_COUNTER = 0

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #(42, 48)
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #(36, 42)

    (mouth_start, mouth_end) = (48, 68)
    (nose_start, nose_end) = (27, 35)

    use_gpu = torch.cuda.is_available()
    print('use_gpu : ',use_gpu)
    img_size = 320
    conf_thres = 0.25
    iou_thres = 0.45
    device = ''
    update = True
    # Load model yolo
    print('=================Loading models=================')
    t1 = time.time()
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model nhan dien container
    t1 = time.time()
    weights = 'best_5s_60_27e.pt'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    if half:
        model.half()
    t2 = time.time()
    print('time load model yolo : ', t2-t1)


    #########################################
    path_video = "video_ir.mp4"#'rtsp://admin:admin123@192.168.6.48:554/cam/realmonitor?channel=2&subtype=0' #'./video_ir.avi'
    cap = cv2.VideoCapture(path_video)
    # cap = Camera(path_video)
    width, height = (0, 0)
    if (cap.isOpened() == False): 
        print("Error reading video file")
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    frame_width = 1280 #int(cap.get(3))
    frame_height = 720#int(cap.get(4))
    size = (int(frame_width/2), int(frame_height/2))
    result = cv2.VideoWriter('output.avi',\
                            cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    count_f = 1   
    count = 1
    while(True):
        print('------------------------')
        print(frame_width,frame_height )
        print('------------------------')
        ret, frame = cap.read()
        # frame = cap.getFrame()
        # ret=True
        frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2))) 
        t1 = time.time()
        
        if ret:
            t1 = time.time()
            frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.4, img_size = 320) 
            cv2.imshow("demo", frame)
            # result.write(frame)
            t2 = time.time()
            print('frame per second : ', round(t2-t1, 4))
            # cap.seek_to_end()
            count_f+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        if count_f%2==0:
            continue
        
