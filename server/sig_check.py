import cv2
import numpy as np
import datetime
from OCR import OCR
from funct import nearest_TLpoint, nearest_SLpoint, haversine

def wrapping(w, h):
    # 버드아이뷰 조절
    real_x = 10
    real_x2 = 100
    real_y = 100
    src = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
    dst = np.float32([[-0.5 * real_x, 0], [0.5 * real_x, 0], [-0.5 * real_x2, real_y], [0.5 * real_x2, real_y]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    return M

# if Red and (0.05 > TL_Distance):
    # if np.size(Cars) > 2:
    #     row, col = img.shape
    #     for x, y in Cars[1:, :]:
    # #         T = M @ np.array([x, y, 1])
    # #         BE_x = T[0] / T[2]
    # #         BE_y = T[1] / T[2]
    # #         BE_d = (BE_x ** 2 + BE_y ** 2) ** 0.5
    #         # 0+ row * 0.2 < x < row *0.8 and y<S_Line[1]
    #         # if (S_Line[0] < x < S_Line[2]) and (y < S_Line[1]):
    #
    #         # # 크롤링으로 현재 위경도 받아와야됨
    #         # closest_point, S_LINE_Distance = nearest_SLpoint(current_latitude, current_longitude)
    # if (row * 0.3 < x < row * 0.7) and (
    #         BE_d > 1000 * S_LINE_Distance):  # S_LINE_Distance 값은 GPS에서 받은 내위치와 정지선 사이의 거리 값 10/22
    #     if OCR_IMG_CHECK == 0:
    #         # Plate 검출
    #         check_length = 1000
    #         for x_p, y_p, w_p, h_p in Plate[1:, :]:
    #             check = np.sqrt((x - x_p) ^ 2 + (y - y_p) ^ 2)
    #             if (check < check_length):
    #                 check_length = check
    #                 ILG_Plate = [x_p, y_p, w_p, h_p]
    #         OCR_IMG = img
    #         OCR_IMG_CHECK = 1
    #
    #     if Arrow:
    #         left_stack = np.append(left_stack, x)
    #         if np.size(left_stack) > 20:
    #             if np.average(left_stack) - 0.1 * w < x < np.average(left_stack) + 0.1 * w:
    #                 Illegal = 0
    #             else:
    #                 Illegal = 1
    #     else:
    #         Illegal = 1

def sig_detect(OCR_IMG, Illegal, check_OCR, ILG_Plate, ILG_CAR):
    # OCR 판단해서 실행
    if Illegal == 1 and check_OCR == 0:
        # check_length = 1000
        # for x_p, y_p, w_p, h_p in Plate[1:, :]:
        #     check = np.sqrt((ILG_CAR[0] - x_p) ^ 2 + (ILG_CAR[1] - y_p) ^ 2)
        #     if (check < check_length):
        #         check_length = check
        #         ILG_Plate = [x_p, y_p, w_p, h_p]
        x_p = ILG_Plate[0]
        y_p = ILG_Plate[1]
        w_p = ILG_Plate[2]
        h_p = ILG_Plate[3]
        plat_result = OCR(OCR_IMG, x_p, y_p, w_p, h_p)
        Illegal = 0
        check_OCR = 1
        return plat_result, Illegal, check_OCR


