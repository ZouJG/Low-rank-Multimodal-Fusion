import cv2
import os
dir_name = "/home/tarena/桌面/视频"
out_dir = "/home/tarena/桌面/视频取帧"

for video in os.listdir(dir_name):
    video_path = os.path.join(dir_name,video)
    ID = video.split('_')[0]
    frames_dir = os.path.join(out_dir,ID)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    framewidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(fps)
    print(totalFrameNumber)
    print(frameheight)
    print(framewidth)
    start = 500
    # videotime = totalFrameNumber / fps
    count = 0
    catchframes = 100
    framegap = totalFrameNumber // catchframes
    print(framegap)
    while count < totalFrameNumber:
        success, frame = cap.read()
        # framegap = int((totalFrameNumber / catchframes)/fps*1e3)
        # print(framegap)
        if count % framegap == 0:
            frame = frame[:, start:start + 1000]
            cv2.imwrite(frames_dir + '/' + str(count) + '.jpg', frame)
            # cv2.waitKey(framegap)
        count += 1

cap.release()
