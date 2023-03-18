# -*- coding: utf-8 -*-

import cv2


def make_aruco_board_image(width,
                           height,
                           size,
                           pixels):

    # These two define the size of the marker. So, its a 4x4 marker, with 1 pixel border = 6x6.
    ar = cv2.aruco.DICT_4X4_50  # Aruco dictionary we will use.
    size_in_bits = 4            # The size_in_bits variable must match the size of dictionary, i.e. DICT_4X4_50.
                                # So, if you change the dictionary, you must change size_in_bits to match.
    border_bits = 1             # This gives the number of pixels in the margin around the tag code. i.e. black border.
    size_of_marker_in_bits = size_in_bits + (2 * border_bits)
    gap_between_markers_in_bits = 2
    gap_ratio = float(gap_between_markers_in_bits) / float(size_of_marker_in_bits)

    # This is the number of markers on the board.
    markers_w = width   # Number of markers in the X direction.
    markers_h = height  # Number of markers in the y direction.

    # This gives the physical units.
    # So when you calibrate, you have the same units.
    marker_length = size  # length of apps marker (millimetres).
    marker_separation = (int)(marker_length * gap_ratio)

    # Create an apps Board (The ids in ascending order starting on 0)
    aruco_dict = cv2.aruco.Dictionary_get(ar)
    grid_board = cv2.aruco.GridBoard_create(markers_w,
                                            markers_h,
                                            marker_length,
                                            marker_separation,
                                            aruco_dict)

    pixels_per_bit = pixels
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

    print(f"Generated image, size={width_pixels} x {height_pixels} pixels.")
    print(f"Load into GIMP and set image size to: {width_millimetres} x {height_millimetres} millimetres when printing.")

    return img
