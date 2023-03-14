import cv2
import numpy as np
from pathlib import Path
import os 

def aruco_display(corners, ids, rejected, image, rvecs, tvecs):
    if len(corners) > 0:
		
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 3)
            print("[Inference] ArUco marker ID: {}".format(markerID))

        avg_z = np.array(tvecs).mean(axis=0).squeeze()
        print(avg_z)

        # WRITING DISTANCE
        cv2.putText(image, str(int(avg_z[2])),(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)	
 
    return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = aruco_dict_type
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)


    rvecs = []
    tvecs = []   
    if len(corners) > 0:
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_LENGTH, matrix_coefficients,
                                                                       distortion_coefficients)
            
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
            rvecs.append(rvec)
            tvecs.append(tvec)

    frame = aruco_display(corners, ids, rejected_img_points, frame, rvecs, tvecs)
    
    tvecs = np.array(tvecs).mean(axis=0).squeeze()
    rvecs = np.array(rvecs).mean(axis=0).squeeze()

    return frame, rvecs, tvecs


def save_frame(img,frame_num, intrinsics, distortion):
    print('saving frame')
    # UNDISTORT FRAME
    h,  w = img.shape[:2]
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, intrinsics, distortion, None, intrinsics)
    #mapx, mapy = cv2.initUndistortRectifyMap(INTRINSICS, DISTORTION, None, newcameramtx, (w,h), 5)
    #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]

    # undistorted
    cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,frame_num), img)#save_folder, i
    # distorted
    cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder,frame_num), dst)#save_folder, i


if __name__=='__main__':
    project_path = Path(__file__).parent.resolve()
    # folder will be structured as follows:
    # assets/type/folder/images

    type='aruco' # random / phantom / EM_tracker_calib / tests
    folder = 'shelves_2'

    save_folder = f'{project_path}/assets/{type}/{folder}'

    MARKER_LENGTH = 30 # mm
    RECORD_ALL=False
    FRAME_RATE=1
    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')  
        os.makedirs(f'{save_folder}/undistorted')  

    intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')
    

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(0)

    rvecs_all = []
    tvecs_all = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        output, rvecs, tvecs  = pose_estimation(gray, arucoDict, intrinsics, distortion)
        
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print('recording frame')
            save_frame(frame,frame_num, intrinsics, distortion)
            rvecs_all.append(rvecs)
            tvecs_all.append(tvecs)
        
        if RECORD_ALL:
            if frame_num%FRAME_RATE==0:
                if np.isnan(rvecs).all() or np.isnan(tvecs).all():
                    continue
                else:
                    print('recording frame')
                    save_frame(frame,frame_num, intrinsics, distortion)
                    rvecs_all.append(rvecs)
                    tvecs_all.append(tvecs)

        frame_num += 1
        if key == ord("q"):
            break
        

    rvecs_all = np.array(rvecs_all).squeeze() #.mean(axis=0)
    tvecs_all = np.array(tvecs_all).squeeze() #.mean(axis=0)

    # SAVE RT VECTORS AND TIMES
    np.save(f'{save_folder}/rvecs',np.array(rvecs_all))
    np.save(f'{save_folder}/tvecs',np.array(tvecs_all))

        

cv2.destroyAllWindows()
cap.release()


