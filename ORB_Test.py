import cv2 as cv
import numpy as np

# 이미지 불러오기 
image = cv.imread("close_shoes.jpg") # 여기에 사용할 이미지 경로 입력
image = cv.resize(image,(600,800))


# HSV 변환
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


# 회색 대리석 필터링(예상되는 회색 범위 설정)
lower_gray = np.array([0, 0, 80])   #HSV에서 어두운 회색
upper_gray = np.array([180, 50, 220])   # 밝은 회색


# 회색 영역 마스크 생성
gray_mask = cv.inRange(hsv, lower_gray, upper_gray)


# 마스크 반전 (회색이 아닌 부분만 남기기)
non_gray_mask = cv.bitwise_not(gray_mask)


# 원본 이미지에서 회색 제거 
filtered_image = cv.bitwise_and(image, image, mask=non_gray_mask)



# 흑백변환 
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # ORB는 흑백 이미지를 사용함(그레이스케일 변환)
gray = cv.cvtColor(filtered_image, cv.COLOR_BGR2GRAY)   # 그레이스케일 변환


# 이미지 대비 향상(CLAHE 적용)
# clahe = cv.createCLAHE(clipLimit = 3.5, tileGridSize = (8,8)) 
# gray = clahe.apply(gray)


# 노이즈 제거를 위한 GaussianBlur 적용 
gray_blur = cv.GaussianBlur(gray,(5, 5),0)



# 대비 향상을 위한 Adaptive Threshold 적용 
# adaptive_thresh = cv.adaptiveThreshold(
#     gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
# )


# Conny 엣지 검출적용 (경계선 강조) 
edges = cv.Canny(gray_blur, 100, 200)


# ORB 객체 생성 
orb = cv.ORB_create(nfeatures = 1500, WTA_K = 4, edgeThreshold = 10, patchSize = 30) # 기본값은 500



# 특징점 검출 
keypoints, descriptors = orb.detectAndCompute(edges, None)


# 특징점 필터링(반응값 기준)
filtered_keypoints = [kp for kp in keypoints if 0.05 < kp.response < 0.2]


# 특징점 그리기
# output_image = cv.drawKeypoints(image, filtered_keypoints, None, (0, 255, 0), flags = 0)
output_image = cv.drawKeypoints(filtered_image, filtered_keypoints, None, (0, 255, 0), flags = 0)
# output_image = cv.drawKeypoints(image, keypoints, None, (0, 255, 0), flags = 0)

# 결과 출력 
while True:
    cv.imshow("ORB Keypoints (Filtered)", output_image)
    cv.imshow("Canny Edge (Filtered)", edges)

    if cv.waitKey(1) & 0xFF == ord('q'):   # 프로그램 종료
        break

        

cv.destroyAllWindows()