import cv2
cap=cv2.VideoCapture(0)
i=1993
while True:
    ret,img=cap.read()
    cv2.imshow('webcam',img)
    k=cv2.waitKey(10)
    if k==27:
        break;
    if k == 13:
        path = f'{i}'.zfill(4) + ".jpg"
        # cv2.imwrite('../../MGR/chessCV/example_img.jpg',img[80:440,100:540])
        i += 1

cap.release()
cv2.destroyAllWindows()

# img = cv2.imread("image.jpg")
# img2 = img[80:440,100:540]
# cv2.imshow("window",img2)
# print(img2.shape)
# cv2.waitKey(0)
