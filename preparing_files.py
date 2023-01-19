import numpy as np
from pathlib import Path
import os
import glob


if __name__=='__main__':
    RENAME = False

    project_path = Path(__file__).parent.resolve()

    # path of image sequence
    #vid_paths = sorted(glob.glob(f'{project_path}/assets/endo_sequence/seq_1/*.*'))
    images_path = f'{project_path}/assets/phantom/surface/images'
    vid_paths = sorted(glob.glob(f'{images_path}/*.*'))

    # RENAMING TO 8 NUMBER FORMAT 
    
    if len(vid_paths[0].split('/')[-1])<11:
        print('changing file names')
        for idx in range(0,len(vid_paths)):
            old_pth = vid_paths[idx] 
            frame_num = int(old_pth.split('/')[-1].split('.')[0]) # name of file then number of frame
            # Absolute path of a file
            #old_name = r"E:\demos\files\reports\details.txt"
            new_pth = '{}/{:08d}.png'.format(images_path, frame_num)

            # Renaming the file
            os.rename(old_pth, new_pth)
    
    
    #f = open("assets/endo_pairs.txt","w+")
    f = open("assets/phantom_surface_pairs.txt","w+")
    for idx in range(0,len(vid_paths)-1):
        pth_1 = vid_paths[idx].split('/')[-1]
        pth_2 = vid_paths[idx+1].split('/')[-1]

        str = f'{pth_1} {pth_2} 0 0'
        f.write(str)
        f.write('\n')
    
    f.close()
    