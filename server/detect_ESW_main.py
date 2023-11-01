# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import cv2
import numpy as np
import math
from pathlib import Path
import torch
import pytesseract
import datetime
import pandas as pd
import glob

from urllib.request import urlopen
from bs4 import BeautifulSoup

import sig_check

web="https://37f3-203-249-81-127.ngrok-free.app/gps"

df1 = pd.read_csv('C:/yolov5-master/yolov5-master/Trafficlight.csv')
df2 = pd.read_csv('C:/yolov5-master/yolov5-master/Stop_Line.csv')
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #colab은 이 코덱만 지원한다고 함
record = True

#colab 드라이브 마운트
# from google.colab import drive
# drive.mount(/content/drive)

#보조 기능들 불러오기
from funct import haversine, gps, nearest_SLpoint, nearest_TLpoint, split_line
from send_data import save_result_to_drive


from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

def auto_canny(image, sigma=0.6):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    y_stack = np.array([])
    left_stack = np.array([])
    S_Line = np.array([0,0,0,0])
    Red = 0
    Arrow = 0
    Green = 0
    Illegal = 0
    Signal = 999
    detect_class = np.array([[0,0,0,0,0]])
    Plate_class = np.array([0,0,0,0])
    d_check = 0
    ILG_CAR = np.array([0,0])
    ILG_Plate = np.array([0,0,0,0])
    OCR_IMG_CHECK = 0
    unprotected = 0
    save_Illegal = 0

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    check_OCR = 0
    check_record = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'
 
        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(xywh)
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        c, x_now, y_now, w_now, h_now = split_line(list(line))
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if c == 0:
                            Red = 1
                        if c == 1:
                            Arrow = 1
                        if c == 2:
                            Green = 1
                            OCR_IMG_CHECK = 0
                            save_Illegal = 0
                        if c == 3 or c == 4 and d_check == 0:
                            detect_class = c, x_now, y_now, w_now, h_now
                            d_check = 1
                            # Cars = np.concatenate([Cars, np.array([[xywh[0], (xywh[1] - 0.5 * xywh[3])]])])

                        if c == 3 or c == 4 and d_check != 0:
                            detect_class = np.concatenate((detect_class, [c, x_now, y_now, w_now, h_now]), axis = 0)
                            d_check +=1
                            # Plate = np.concatenate([Plate, np.array([[xywh[0], xywh[1], xywh[2], xywh[3]]])])
                        if c == 5:
                            Plate_class = x_now, y_now, w_now, h_now
                        if c == 6:
                            unprotected = 1


                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                #날짜 씌우기
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                cv2.putText(im0, now, (930, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        ## 신호 위반 판단
        # 현재 위경도 좌표
        current_latitude, current_longitude = gps(web)
        # 신호등까지 거리 계속 갱신
        TL_Distance = nearest_TLpoint(current_latitude, current_latitude)
        closest_point, S_LINE_Distance = nearest_SLpoint(current_latitude, current_longitude)
        if Red == 1 and 50 > TL_Distance and unprotected == 0:
            size = np.shape(im0s)
            h = size[1]
            w = size[2]
            M = sig_check.wrapping(w, h)        #원근 변환 행렬 구하기

            img = im0s[0]
            # 이미지에 날짜 씌우기 및 저장
            file_name = now.strftime("%Y%m%d_%H%M") + '.mp4'
            now = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
            cv2.putText(img, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

            #Red시 저장 시작
            # video = cv2.VideoWriter(file_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            # video.write(img)
            check_record += 1
            # 이건 colab 저장시
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('/content/drive/MyDrive/Colab Videos/output_video.avi', fourcc, 30, (640, 480))
            out.write(img)

            if Red == 1:
                row, col = detect_class.shape
                for i in range(0, row):
                    img_x, img_y = img.shape
                    x = detect_class[i][1]
                    y = detect_class[i][2]
                    T = M @ np.array([x, y, 1])
                    BE_x = T[0] / T[2]
                    BE_y = T[1] / T[2]
                    BE_d = (BE_x ** 2 + BE_y ** 2) ** 0.5
                    if (row * 0.3 < x < row * 0.7) and (
                            BE_d > S_LINE_Distance):  # S_LINE_Distance 값은 GPS에서 받은 내위치와 정지선 사이의 거리 값 10/22
                        if OCR_IMG_CHECK == 0:
                            # Plate 검출
                            check_length = 1000
                            for x_p, y_p, w_p, h_p in Plate_class:
                                check = np.sqrt((x - x_p) ^ 2 + (y - y_p) ^ 2)
                                if (check < check_length):
                                    check_length = check
                                    ILG_Plate = [x_p, y_p, w_p, h_p]
                            OCR_IMG = img
                            OCR_IMG_CHECK = 2
                            ILG_CAR[0] = x
                            ILG_CAR[1] = y

                        if Arrow == 1:
                            left_stack = np.append(left_stack, x)
                            if np.size(left_stack) > 20:
                                # if np.average(left_stack) < left_stack[0] #좌회전 조건으로 가도 될거같음
                                if np.average(left_stack) - 0.1 * w < x < np.average(left_stack) + 0.1 * w:
                                    Illegal = 0
                                else:
                                    Illegal = 1
                        else:
                            left_stack = np.array([])
                            Illegal = 1



                    Plate_result, Illegal_1, check_OCR_1 = sig_check.sig_detect(OCR_IMG, Illegal, OCR_IMG_CHECK, ILG_Plate, ILG_CAR)
                    Illegal, OCR_IMG_CHECK = Illegal_1, check_OCR_1

            elif Red != 1 or check_record > 1000 or unprotected == 1:
                # video.release()
                out.release()
                check_record = 0
                OCR_IMG_CHECK = 1
                Illegal = 0

        if Illegal == 1 and save_Illegal == 0:
            result = list(Plate_result + ',' + str(current_latitude) + ',' + str(current_longitude))
            save_result_to_drive(result)
            save_Illegal = 1


        Red = 0
        Arrow = 0
        Green = 0
        Plate_class = np.array([[0, 0, 0, 0]])
        d_check = 0
        unprotected = 0
        detect_class = np.array([[0, 0, 0, 0, 0]])

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
#http://192.168.1.101:8080/stream

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'ESW_best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images_1' , help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/ESW_data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # parser.add_argument("--ip", help="a dummy argument to fool ipython", default="127.0.0.1")
    # parser.add_argument("--stdin", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--control", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--hb", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--Session.key", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--Session.signature_scheme", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--shell", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--transport", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--iopub", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
