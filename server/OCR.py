import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def OCR(image, x_p,y_p,w_p,h_p):
    y_s = round(y_p - 0.5 * h_p + 0.1 * h_p)
    y_end = round(y_p + 0.5 * h_p - 0.1 * h_p)
    x_s = round(x_p - 0.5 * h_p + 0.1 * w_p)
    x_end = round(x_p + 0.5 * h_p - 0.1 * w_p)
    img_ori = image[ y_s:y_end, x_s:x_end ]
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 형태학적 변환 사각형
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)  # 밝기값 크게 변하는 영역 강조
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)  # 어두운 부분 강조

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)  # graycale에 밝기값 변하는 부분 합침
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)  # 픽셀값 뺼셈, 밝기 강조 더한거에 어두운 부분 강조 뻄
    img_blurred = cv2.GaussianBlur(gray, (15, 15), 0)  # ksize = 가우시안 커널 크기, sigmaX = x방향 시그마
    # adaptive Threshold 상황에 따라 적용
    # img_thresh = cv2.adaptiveThreshold(                                             #adaptive Threshold 적용부분
    #     img_blurred,
    #     maxValue = 250.0,                                           #임계함수 최댓값, 보통 255
    #     adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            #가우시안가중치 평균 사용
    #     thresholdType = cv2.THRESH_BINARY_INV,                      #임계값 보타 그면 0, 아니면 value값으로 바꿈(INV안쓴건 반대로)
    #     blockSize = 11,                                              #블록크기, 3이상 홀수
    #     C = 1                                                       #블록 내 가중평균에서 뺼 값
    # )
    # print(img_thresh.shape)
    img_thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    output_plate = pytesseract.image_to_string(img_thresh, lang='kor', config='--psm 7 --oem 0')
    return output_plate
