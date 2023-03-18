'''
this file will create an aruco board of specified dimensions and aruco dict type.
bug: can't figure out how to make image same dimension for printing
'''
import cv2
from pathlib import Path


if __name__=='__main__':
    # https://stackoverflow.com/questions/59789491/using-board-create-for-aruco-markers-leads-to-error-due-to-objpoints-not-in-the

    project_path = Path(__file__).parent.resolve()
    # folder will be structured as follows:
    # assets/type/folder/images

    # Settings for the marker

    # These two define the size of the marker. So, its a 4x4 marker, with 1 pixel border = 6x6.
    ar = cv2.aruco.DICT_4X4_50  # Aruco dictionary we will use.
    size_in_bits = 4            # The size_in_bits variable must match the size of dictionary, i.e. DICT_4X4_50.
                                # So, if you change the dictionary, you must change size_in_bits to match.
    border_bits = 1             # This gives the number of pixels in the margin around the tag code. i.e. black border.
    size_of_marker_in_bits = size_in_bits + (2 * border_bits)
    gap_between_markers_in_bits = 2
    gap_ratio = float(gap_between_markers_in_bits) / float(size_of_marker_in_bits)

    # This is the number of markers on the board.
    markers_w = 5  # Number of markers in the X direction.
    markers_h = 7  # Number of markers in the y direction.

    # This gives the physical units.
    # So when you calibrate, you have the same units.
    marker_length = 30  # length of aruco marker (millimetres).
    marker_separation = (int)(marker_length * gap_ratio)

    # Create an aruco Board (The ids in ascending order starting on 0)
    #aruco_dict = cv2.aruco.Dictionary_get(ar)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ar)  # dictionary of markers provided
    #grid_board = cv2.aruco.GridBoard_create(markers_w,
    #                                        markers_h,
    #                                        marker_length,
    #                                        marker_separation,
    #                                        aruco_dict)
    # creat an aruco Board (The ids in ascending order starting on 0)
    grid_board = cv2.aruco.GridBoard((markers_w, markers_h),
                                     marker_length,
                                     marker_separation,
                                     aruco_dict)
    pixels_per_bit = 10
    width_millimetres = (markers_w * marker_length) + ((markers_w - 1) * marker_separation)
    height_millimetres = (markers_h * marker_length) + ((markers_h - 1) * marker_separation)
    width_pixels = pixels_per_bit * ((markers_w * size_of_marker_in_bits) + ((markers_w - 1) * gap_between_markers_in_bits))
    height_pixels = pixels_per_bit * ((markers_h * size_of_marker_in_bits) + ((markers_h - 1) * gap_between_markers_in_bits))
    img_size = (width_pixels, height_pixels)

    img = cv2.aruco.drawPlanarBoard(board=grid_board,
                                    outSize=img_size,
                                    marginSize=0,
                                    borderBits=border_bits
                                    )

    print(f"Load image into GIMP. Image size={width_pixels} x {height_pixels} pixels.")
    print(f"Set image size to: {width_millimetres} x {height_millimetres} millimetres")
    cv2.imwrite("aruco_board.png", img)

