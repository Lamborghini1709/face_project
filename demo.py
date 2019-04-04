

# coding=utf-8


import cv2
import face
import time
import os
from classifier_job import schedule_jobs
import schedule
import numpy as np
from datetime import datetime
from classifier_job import job1,job2


#人脸画框并显示预测结果
def add_overlays(frame, facename,face):
    face_bb = face.bounding_box.astype(int)
    if facename == 'New customer':
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    cv2.rectangle(frame,
                  (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                  color, 2)
    # print(facename)

    cv2.putText(frame, facename, (face_bb[0], face_bb[3]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                thickness=2, lineType=2)






#判断保存人脸函数
def save_img(img,face_recognition):
    faces, prob = face_recognition.identify(img)
    root_dir = './save_image'
    dir_list = []
    today = datetime.today()

    label_id = int(today.strftime('%Y%m%d') + '1000')
    save_flag = False

    if len(faces) < 1:
        return False,None,None
    else:
        pass

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        save_label = os.path.join(root_dir, str(label_id))
        os.mkdir(save_label)
        savename = os.path.join(save_label, str(time.time()) + '.png')
        cv2.imwrite(savename, img)
        save_flag = True
    else:
        for root, dirs, _ in os.walk(root_dir):
            for dir in dirs:
                # print(file)
                dir_list.append(os.path.join(root, dir))

        if len(dir_list) == 0:
            save_label = os.path.join(root_dir, str(label_id))
            os.mkdir(save_label)
            savename = os.path.join(save_label, str(time.time()) + '.png')
            cv2.imwrite(savename, img)
            save_flag = True

        else:
            for i in dir_list:
                for root, _, files in os.walk(i):
                    if len(files) <= 10:
                        euc_list = []
                        for file in files:
                            # print(file)
                            vs_file = os.path.join(root, file)
                            vs_img = cv2.imread(vs_file)
                            faces_vs, probs = face_recognition.identify(vs_img)
                            if len(faces_vs) >= 1:
                                euc = float(
                                    '%.4f' % np.sqrt(
                                        np.sum(np.square(np.subtract(faces[0].embedding, faces_vs[0].embedding)))))
                                euc_list.append(euc)
                            else:
                                print('已存文件夹照片未检测到人脸',file)
                                continue
                        if np.mean(euc_list) <= 0.8:
                            if len(files) <= 100:
                                savename = os.path.join(i, str(time.time()) + '.png')
                                # print('saved image')
                                save_flag = True
                                return True,savename,img

                            else:
                                save_flag = True
                                print('照片已满')
                                return False,None,None
                        else:
                            print('文件夹名称',i)
                            print('该文件夹不是同一个人')
                            continue

                    else:
                        euc_list = []
                        for file in files[:10]:
                            # print(file)
                            vs_file = os.path.join(root, file)
                            vs_img = cv2.imread(vs_file)
                            save_label = vs_file.split('/')[-2]
                            faces_vs, probs,predictions = face_recognition.identify(vs_img)
                            if len(faces_vs) >= 1:
                                euc = float(
                                    '%.2f' % np.sqrt(
                                        np.sum(np.square(np.subtract(faces[0].embedding, faces_vs[0].embedding)))))
                                print('欧式距离',euc)

                                euc_list.append(euc)
                                # if len(files) <3:
                            else:
                                print('已存文件夹照片未检测到人脸',file)
                                continue
                        if np.mean(euc_list) <= 1:
                            if len(files) <= 100:
                                savename = os.path.join(i, str(time.time()) + '.png')
                                print(savename)

                                # print('saved image')
                                save_flag = True
                                return True,savename,img
                            else:
                                save_flag = True
                                print('照片已满')
                                return False,None,None
                        else:
                            print('文件夹名称', i)
                            print('该文件夹不是同一个人')
                            continue

            if not save_flag:
                save_dir = os.path.join(root_dir, str(max(list(map(int, list(i.split('/')[-1] for i in dir_list)))) + 1))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                savename = os.path.join(save_dir, str(time.time()) + '.png')
                print(savename)

                # print('saved image')
                save_flag = True
                return True, savename, img

#保存人脸函数
def save_img_func(img,face_recognition):
    flag,savename,img = save_img(img,face_recognition)
    if flag:
        cv2.imwrite(savename,img)
        print('saved image')
    else:
        pass

def main():
    detect_flag_list = []
    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    schedule.every().day.at("00:30").do(schedule_jobs)

    count = 0
    prob_list = []
    change_flag = False

    while True:

        # print('*'*40)
        count += 1
        if count % 100 == 10:
            change_flag = True
            prob_list = []
        # print(count)
        # prob_list = []
        schedule.run_pending()
        ret, frame = video_capture.read()

        faces,prob,predictions = face_recognition.identify(frame)
        # print(predictions)
        prob_list.append(prob)

        #每100次统计一次概率最大值
        # print(prob_list)
        # print(len(prob_list) % 100)
        if prob_list and len(prob_list) % 100 == 0:
            # add_overlays(frame,faces[0].name,faces[0])
            # print('*'*40)
            print(float(np.max(prob_list)))

        # print('概率值',prob)
        img = frame.copy()
        if len(faces) >= 1:
            # if prob > np.sum(predictions[0][-3:-1]):
            if prob > 0.6:
                detect_flag = 1
                facename = faces[0].name
                # facename = 'Old member'
                # change_flag = False
                # prob_list = []
                # print(detect_flag)
            else:
                detect_flag = 2
                # detect_flag_list.append(detect_flag)
                # if change_flag:
                facename = 'New customer'
                # change_flag = False
                # prob_list = []
                # print(detect_flag)

            add_overlays(img, facename,faces[0])
        else:
            detect_flag = 0
            # detect_flag_list = []
            # print(detect_flag)


        # try:
        #     if len(detect_flag_list) >= 10:
        #         save_img_func(frame, face_recognition)
        #         # print('saved image')
        # except:
        #     print('检测异常')


        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # job1()
    # job2()
    main()