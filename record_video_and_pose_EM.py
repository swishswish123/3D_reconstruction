from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


'''
def record_images():
    print('press c to capture image and q to quit')
    # define a video capture object
    vid = cv2.VideoCapture(0)
    i = 0

    while(True):
        i +=1
        k = cv2.waitKey(33)

        # Capture the video frame by frame
        ret, frame = vid.read()

        # UNDISTORT FRAME
        dst = cv2.undistort(frame, INTRINSICS, DISTORTION, None, INTRINSICS)
        
        cv2.imshow(f'images', dst)
        
        if k == ord('c') or CAPTURE_ALL: # or i%FRAME_RATE==0:
            print('capturing frame a')
            
            # SAVE FRAME
            cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,i), frame)#save_folder, i
            # SAVE UNDISTORTED FRAME
            cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder,i), dst)

        if cv2.waitKey(1) & 0xFF == ord('q') or i==MAX_FRAMES:
            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            break
'''

def record_RT_vecs():

    # define a video capture object
    vid = cv2.VideoCapture(0)

    i = 0
    if TRACKING:
        # Where we will save 4x4 matrices and times
        RT_vecs = []
        times = []

        points = []
        vectors = []

        # point at origin of tracker
        original_point = np.array([0,0,0,1])
        unit_vec = np.array([1,0,0,0])

    while(True):
        k = cv2.waitKey(33)

        # Capture the video frame by frame
        ret, frame = vid.read()

        # UNDISTORT FRAME
        dst = cv2.undistort(frame, INTRINSICS, DISTORTION, None, INTRINSICS)  
        cv2.imshow(f'images', dst)

        # current frame time
        if i ==0:
            print("recording....")
            start_time = time.time()
            time_diff = 0
        else:
            time_diff =  time.time()-start_time
        
        if k == ord('c') or CAPTURE_ALL: # or i%FRAME_RATE==0:
            print('capturing frame a')
            
            port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        
            # SAVE FRAME
            cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,i), frame)#save_folder, i
            # SAVE UNDISTORTED FRAME
            cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder,i), dst)

            # SAVE 4X4 MATRICES
            if TRACKING:
                RT_vecs.append(tracking[0])
                times.append(time_diff)
                print(tracking[0])

                # PLOT POINT!!!!!
                point = tracking[0]@original_point
                #ax.scatter(point[0],point[1],point[2])
                vector = tracking[0] @ unit_vec
                
                points.append(point)
                vectors.append(vector)

        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q') or i==MAX_FRAMES:
            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            break

        i+=1
    
    if TRACKING:
        return RT_vecs, times, np.array(points), np.array(vectors)
    else:
        return



if __name__ == "__main__":
    project_path = Path(__file__).parent.resolve()
    
    # folder will be structured as follows:
    # assets/type/folder/images

    type='tests' # random / phantom / EM_tracker_calib / tests
    folder = 'aruCo'

    save_folder = 'f{project_path}/assets/{type}/{folder}'
    
    # Capture_all:
    # True will capture every single frame (with frame rate specified)
    # False: you will have to press the key 'c' to capture a frame
    CAPTURE_ALL = False
    # how often to record frames- only applies if CAPTURE_ALL is true
    FRAME_RATE = 10

    # max frames that will be recorded
    MAX_FRAMES = 300

    # True: tracking data from EM tracker will be recorded
    # False: tracking data will not be recorded
    TRACKING = False

    # where intrinsics and distortion are located
    INTRINSICS = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    DISTORTION =  np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')


    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')  
        os.makedirs(f'{save_folder}/undistorted')
    
    # records images and tracking info
    if TRACKING:
        # init params
        SETTINGS = {
            "tracker type": "aurora",
                }
        TRACKER = NDITracker(SETTINGS)
        TRACKER.start_tracking()
        
    # recordinng frames and Rt vectors
    RT_vecs, times, points, vectors = record_RT_vecs()
    

    if TRACKING:
        # visualising tracked camera positions
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((5, 1,5))
        ax.quiver(points[:,0],points[:,1],points[:,2], vectors[:,0], vectors[:,1], vectors[:,2], length=5, normalize=True, linewidths=3)
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y',fontsize=20)
        ax.set_zlabel('Z', fontsize=30, rotation=60)
        plt.show()
    
        # SAVE RT VECTORS AND TIMES
        np.save(f'{save_folder}/vecs',np.array(RT_vecs))
        np.save(f'{save_folder}/times',np.array(times))

        np.save(f'{save_folder}/points',points)
        np.save(f'{save_folder}/vectors',vectors)

        TRACKER.stop_tracking()
        TRACKER.close()
    



   