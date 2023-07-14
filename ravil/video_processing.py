from os.path import join as pjoin
import os
import cv2

def get_first_frame(videopath=pjoin('..','Test1 - alecsandr27000.mkv') ,framepath=None):
    video = cv2.VideoCapture(videopath)
    # pjoin('..','..','Test1 - alecsandr27000.mkv')
    _, frame_0 = video.read()
    # np.save('frame_0.npy', frame0)
    video.release()
    del video
    if framepath:
        cv2.imwrite(framepath, frame_0)
    return frame_0