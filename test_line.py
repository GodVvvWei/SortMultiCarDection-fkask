import cv2

line = [(0, 150), (1200, 150)]
vc = cv2.VideoCapture('1.mp4')

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
while rval:
    grabed, frame = vc.read()
    if not grabed:
        break
    cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(20)