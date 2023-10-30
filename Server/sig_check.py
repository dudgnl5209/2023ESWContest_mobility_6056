import cv2
import numpy as np
from OCR import OCR

def wrapping(w, h):
    # 버드아이뷰 조절
    real_x = 10
    real_x2 = 100
    real_y = 100
    src = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
    dst = np.float32([[-0.5 * real_x, 0], [0.5 * real_x, 0], [-0.5 * real_x2, real_y], [0.5 * real_x2, real_y]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    return M

def sig_detect(OCR_IMG, Illegal, check_OCR, ILG_Plate, ILG_CAR):
    # OCR 판단해서 실행
    if Illegal == 1 and check_OCR == 2:
        x_p = ILG_Plate[0]
        y_p = ILG_Plate[1]
        w_p = ILG_Plate[2]
        h_p = ILG_Plate[3]
        plat_result = OCR(OCR_IMG, x_p, y_p, w_p, h_p)
        Illegal = 1
        check_OCR = 1
        return plat_result, Illegal, check_OCR

    else:
        return 'None', Illegal, check_OCR


