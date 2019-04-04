#coding=utf-8
import cv2
print(2)
mp4_file = ["./video/34.mp4","./video/35.mp4","./video/36.mp4"]
c = 1
for i in mp4_file:
    vc = cv2.VideoCapture(i)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        cv2.imwrite('./video_image/chengcan/' + str(c) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)
    vc.release()