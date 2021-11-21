import sys
import os
import cv2
import glob
path = 'S:/media/data1/mhkim/SR/face_video'
save_path = 'S:/media/data1/mhkim/SR/face_video_frame'
list_mp = glob.glob(os.path.join(path,'*'))

swit = True
for _mp in list_mp:
    print(_mp)

    count = 0
    title = _mp.split('\\')[-1]
    _save_path = os.path.join(save_path,title)
    try:
        os.makedirs(_save_path,exist_ok=False)
    except:
        continue
    
    vidcap = cv2.VideoCapture(_mp)
    ret = True
    while(ret):
    #while(vidcap.isOpened()):
        ret, image = vidcap.read() # 이미지 사이즈 960x540으로 변경
        # image = cv2.resize(image, (960, 540)) # 30프레임당 하나씩 이미지 추출
        if not ret: break
        if(int(vidcap.get(1)) % 10 == 0):
            # print('Saved frame number : ' + str(int(vidcap.get(1)))) # 추출된 이미지가 저장되는 경로
            cv2.imwrite(_save_path+f"/{title}_{count}.png", image) #print('Saved frame%d.jpg' % count)
            count += 1

    #vidcap.release()
    #cv2.destroyWindow
print("END")
cv2.destroyAllWindows()

