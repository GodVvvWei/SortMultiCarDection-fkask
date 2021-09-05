

#作者：小约翰啊伟
#B站主页：https://space.bilibili.com/420694489
#源码下载：https://github.com/GodVvvWei/SortMultiCarDection-fkask

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from flask import Flask,render_template,Response
import cv2

from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from kalman import *

app = Flask(__name__)
user, pwd, ip = "admin", "123456zaQ", "[192.168.100.196]"
from yolov5_detect import detect



# 线与线的碰撞检测：叉乘的方法判断两条线是否相交
# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流（海康摄像头）

        # self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, 1))
        self.video = cv2.VideoCapture('./input/1.mp4')
        #大华摄像头
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))

        self.weights, imgsz = 'yolov5s.pt', 640
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # 车辆总数
        self.counter = 0
        # 正向车道的车辆数据
        self.counter_up = 0
        # 逆向车道的车辆数据
        self.counter_down = 0

        # 创建跟踪器对象
        self.tracker = Sort()
        self.memory = {}

        self.line = [(0, 200), (1800, 200)]
    def __del__(self):
        self.video.release()

    def get_frame(self):

        # for i in range(50):
        success, frame = self.video.read()

        Outputs= detect(source=frame,half=self.half,model=self.model,device=self.device,imgsz=self.imgsz,stride=self.stride)
        dets = Outputs[:, :4].numpy()
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        # SORT目标跟踪
        if np.size(dets) == 0:
            return frame
        else:
            tracks = self.tracker.update(dets)
        # 跟踪框
        boxes = []
        # 置信度
        indexIDs = []
        # 前一帧跟踪结果
        previous = self.memory.copy()
        self.memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            self.memory[indexIDs[-1]] = boxes[-1]

            # 生成多种不同的颜色
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
            # 碰撞检测
            if len(boxes) > 0:
                i = int(0)
                # 遍历跟踪框
                for box in boxes:
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
                    color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                    cv2.rectangle(frame, (x, y), (w, h), color, 2)

                    # 根据在上一帧和当前帧的检测结果，利用虚拟线圈完成车辆计数
                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                        p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))

                        # 利用p0,p1与line进行碰撞检测
                        if intersect(p0, p1, self.line[0], self.line[1]):
                            self.counter += 1
                            # 判断行进方向
                            if y2 > y:
                                self.counter_down += 1
                            else:
                                self.counter_up += 1
                    i += 1

            # 将车辆计数的相关结果放在视频上
            cv2.line(frame, self.line[0], self.line[1], (0, 255, 0), 3)
            cv2.putText(frame, 'counter:' + str(self.counter), (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 3)
            cv2.putText(frame, 'counter_up:' + str(self.counter_up), (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, 'counter_down:' + str(self.counter_down), (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 0, 255), 3)

        ret,jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


@app.route('/xyhaw')
def xyhaw():
    return render_template('xyhaw.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
