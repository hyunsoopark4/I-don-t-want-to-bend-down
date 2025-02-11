import cv2


cap = cv2.VideoCapture(0)
sample = cv2.imread("many_shoes.jpg")
trans = cv2.resize(sample,(300,400))

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break
    
    cv2.imshow("Cam1",frame)

    gray = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Edges",edges)

    if cv2.waitKey(1) & 0xFF == ord('s'):   # 저장 기능
        cv2.imwrite("shoe_image.jpg",frame)
        cv2.imwrite("edge_image.jpg",edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # 프로그램 종료
        break

cap.release()
cv2.destroyAllwindows()