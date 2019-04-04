# coding=utf-8


import cv2
import face
import time
import os
from img_vs_folder import save_or_not
from classifier_job import schedule_jobs
import tensorflow as tf
import facenet
import schedule



#save unkonw customer face
def save_img(frame,face_id):
    save_img_rootdir = 'save_image'
    save_img_dir = os.path.join(save_img_rootdir, face_id)
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    filename = os.path.join(save_img_dir, str(time.time())+'.png')
    cv2.imwrite(filename,frame)
    print('saved images')



#人脸画框并显示预测结果
def add_overlays(frame, faces,prob,frame_rate):
    print(len(faces))
    if len(faces) >= 1:     #is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            print(prob)
            if  prob <= 0.6:
                face.name = 'new customer'

            print(face.name)
            cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2, lineType=2)

    elif len(faces) == 0:
        print('未检测到人脸')

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=2, lineType=2)


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    prob_list = []
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    face_recognition = face.Recognition()
    save_flag = False
    save_count = 0
    schedule.every().day.at("00:30").do(schedule_jobs)

    while True:
        # Capture frame-by-frame
        schedule.run_pending()
        ret, frame = video_capture.read()
        # print(frame_count)
        if (frame_count % frame_interval) == 0:
            faces,prob = face_recognition.identify(frame)
            #add prob to prob_listq
            prob_list.append(prob)
            print(len(prob_list))
            # if len(faces) == 1 and len(prob_list) >= 100 and max(prob_list) < 0.5:
            #     if save_count == 200:
            #         save_count = 0
            #     save_img(frame,save_img_rootdir,face_id)
            #     save_count += 1
            #     face_id += 1q
            if prob >= 0.5 or len(faces) == 0:
                prob_list = []
            # Check our current fps
            if len(faces) == 1 and len(prob_list) >= 30 and max(prob_list) < 0.5:
                print('**************************')
                save_flag = True
                # save_count = 0
                # result,save_label = save_or_not(frame,face_recognition)
                # if result:
                #     save_img(frame,save_label)
                #     prob_list = []
                # else:
                #     pass
            if save_flag :
                result, save_label = save_or_not(frame, face_recognition)
                if result:
                    save_img(frame, save_label)
                    save_count += 1
                    # prob_list = []
                else:
                    pass
            if save_count == 100:
                save_flag = False
                save_count = 0



            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces,prob,frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()