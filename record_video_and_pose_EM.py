from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path



def record_images():
    print('press c to capture image and q to quit')
    # define a video capture object
    vid = cv2.VideoCapture(0)
    i = 0

    while(True):
        i +=1
        k = cv2.waitKey(33)

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # UNDISTORT FRAME

        img = frame
        h,  w = img.shape[:2]
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(INTRINSICS, DISTORTION, (w,h), 1, (w,h))
        # undistort
        dst = cv2.undistort(img, INTRINSICS, DISTORTION, None, INTRINSICS)
        #mapx, mapy = cv2.initUndistortRectifyMap(INTRINSICS, DISTORTION, None, newcameramtx, (w,h), 5)
        #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
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

def record_RT_vecs():

    # Where we will save 4x4 matrices and times
    RT_vecs = []
    times = []

    # define a video capture object
    vid = cv2.VideoCapture(0)

    i = 0

    points = []
    vectors = []

    # point at origin of tracker
    original_point = np.array([0,0,0,1])
    unit_vec = np.array([1,0,0,0])

    while(True):
        k = cv2.waitKey(33)

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # UNDISTORT FRAME
        img = frame
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(INTRINSICS, DISTORTION, (w,h), 1, (w,h))
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(INTRINSICS, DISTORTION, None, newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
            
        cv2.imshow(f'images.jpg', dst)

        # current frame time
        if i ==0:
            print("recording....")
            start_time = time.time()
            time_diff = 0
        else:
            time_diff =  time.time()-start_time
        
        if k == ord('c'):
            print('capturing frame a')
            
            port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        
            # SAVE FRAME
            #cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,i), frame)#save_folder, i
            cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,i), dst)#save_folder, i
            #cv2.imshow(f'images/{i}.jpg', frame)

            # SAVE 4X4 MATRICES
            RT_vecs.append(tracking[0])
            times.append(time_diff)
            print(tracking[0])

            # PLOT POINT!!!!!
            point = tracking[0]@original_point
            #ax.scatter(point[0],point[1],point[2])
            vector = tracking[0] @ unit_vec
            
            points.append(point)
            vectors.append(vector)

        # the 'q' button is set as the
        # quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q') or i==MAX_FRAMES:
            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            break

        if k == ord('a'):
            print('pressend a')
            break
        elif  k==27:
            print('pressed esc')
            break
        i+=1
    
    return RT_vecs, times, np.array(points), np.array(vectors)



if __name__ == "__main__":
    project_path = Path(__file__).parent.resolve()
    
    # CHANGE ME:
    type='tests' # random / phantom / EM_tracker_calib / tests
    folder = 'aruCo'

    save_folder = 'f{project_path}/assets/{type}/{folder}'
    #save_undistorted_folder = f'{project_path}/assets/{type}/{folder}/undistorted'
    
    MAX_FRAMES = 300
    CAPTURE_ALL = False
    TRACKING = False

    INTRINSICS = np.loadtxt('calibration/mac_calibration/intrinsics.txt')
    DISTORTION =  np.loadtxt(f'calibration/mac_calibration/distortion.txt')

    FRAME_RATE = 10

    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')  
        os.makedirs(f'{save_folder}/undistorted')
    
    record_images()


    if TRACKING:
        SETTINGS = {
            "tracker type": "aurora",
                }
        TRACKER = NDITracker(SETTINGS)
        TRACKER.start_tracking()
        RT_vecs, times, points, vectors = record_RT_vecs()
    

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((5, 1,5))
        #ax.axis('tight')
        ax.quiver(points[:,0],points[:,1],points[:,2], vectors[:,0], vectors[:,1], vectors[:,2], length=5, normalize=True, linewidths=3)
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y',fontsize=20)
        ax.set_zlabel('Z', fontsize=30, rotation=60)
        #plt.autoscale(enable=False)
        plt.show()
    
        print(len(times))

        # SAVE RT VECTORS AND TIMES
        np.save(f'{save_folder}/vecs',np.array(RT_vecs))
        np.save(f'{save_folder}/times',np.array(times))

        np.save(f'{save_folder}/points',points)
        np.save(f'{save_folder}/vectors',vectors)

        TRACKER.stop_tracking()
        TRACKER.close()



    """

    SETTINGS = {
        "tracker type": "aurora",
            }
    TRACKER = NDITracker(SETTINGS)


    TRACKER.start_tracking()


    RT_vecs = []
    for i in range(50):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        
    
        for t in tracking:
            RT_vecs.append(t)
            print(t)
                
    
            
    np.save('vecs',np.array(RT_vecs))

    TRACKER.stop_tracking()
    TRACKER.close()
    """