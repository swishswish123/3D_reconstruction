'''
this file will create an aruco board of specified dimensions and aruco dict type.
bug: can't figure out how to make image same dimension for printing
'''
import cv2
from pathlib import Path


def generate_aruco_board_for_printing(aruco_dict=cv2.aruco.DICT_4X4_50,
                                      size_in_bits=4,
                                      border_bits=1,
                                      gap_between_markers_in_bits=2,
                                      marker_length=30,
                                      markers_w=5,
                                      markers_h=8,
                                      pixels_per_bit=10
                                      ):
    """ Function that generates aruco markers

    this function will generate an aruco board with the given parameters. The board will be saved in
    the main directory and can be printed after correct size is specified with gimp. The size is printed
    when running the function.

    Args:
        aruco_dict: aruco dictionary that will be used (eg. aruco.DICT_4X4_50)
        size_in_bits: The size_in_bits variable must match the size of dictionary, i.e. DICT_4X4_50.
        border_bits: The black border surrounded by the aruco marker. Eg. if it is 1 and the aruco itself is
                     4x4, then the aruco will now be 6x6 as the black border goes all around the marker.
        gap_between_markers_in_bits: This will be the white gap (in bits) between consecutive aruco markers.
        marker_length: length of each marker in mm
        markers_w: number of aruco markers along the width of the grid
        markers_h: number of aruco markers along the height of the grid
        pixels_per_bit: ################### 

    Returns:
        saves image of aruco board under current project path
    """

    project_path = Path(__file__).parent.resolve()

    # size of marker in bits- where bits refers to the pixel-like shape of the aruco.
    # Each aruco will be composed of the size_in_bits (which is dependent on the dictionary used)
    # plus the number of bits in the border specified. This border is the black border around the marker.
    # For example, if we use the 50 4x4 dict, each marker's size is 4 bits, and if we specify the
    # border_bit to 1, the size in bits will be 6 as there will be a border on each side of the marker.
    size_of_marker_in_bits = size_in_bits + (2 * border_bits)

    # gap between consecutive markers
    gap_ratio = float(gap_between_markers_in_bits) / float(size_of_marker_in_bits)

    # This gives the physical units.
    # So when you calibrate, you have the same units.
    marker_separation = (int)(marker_length * gap_ratio)

    # load aruco dictionary to be used for board
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)  # dictionary of markers provided
    # Create an aruco Board (The ids in ascending order starting on 0)
    grid_board = cv2.aruco.GridBoard((markers_w, markers_h),
                                     marker_length,
                                     marker_separation,
                                     aruco_dict)

    # width will be the number of markers*length plus whatever separation there is between each marker.
    # note that the separation is only within the grid so the last separation isn't counted
    width_millimetres = (markers_w * marker_length) + ((markers_w - 1) * marker_separation)
    # the same applies to the height
    height_millimetres = (markers_h * marker_length) + ((markers_h - 1) * marker_separation)

    # measuring width and height of img in pixels
    width_pixels = pixels_per_bit * (
            (markers_w * size_of_marker_in_bits) + ((markers_w - 1) * gap_between_markers_in_bits))
    height_pixels = pixels_per_bit * (
            (markers_h * size_of_marker_in_bits) + ((markers_h - 1) * gap_between_markers_in_bits))
    img_size = (width_pixels, height_pixels)

    # generating image from board for printing
    img = grid_board.generateImage(img_size,  # outSize size of the output image in pixels.
                                   marginSize=0,  # minimum margins (in pixels) of the board in the output image
                                   borderBits=border_bits  # borderBits width of the marker borders.
                                   )

    print(f"Load image into GIMP. Image size={width_pixels} x {height_pixels} pixels.")
    print(f"Set image size to: {width_millimetres} x {height_millimetres} millimetres")
    cv2.imwrite(f"{project_path}/aruco_board.png", img)


def main():
    # https://stackoverflow.com/questions/59789491/using-board-create-for-aruco-markers-leads-to-error-due-to-objpoints-not-in-the

    # Settings for the marker itself
    # These two define the size of the marker. So, its a 4x4 marker, with 1 pixel border = 6x6.
    ar = cv2.aruco.DICT_4X4_50  # Aruco dictionary we will use.
    size_in_bits = 4  # The size_in_bits variable must match the size of dictionary, i.e. DICT_4X4_50.
    # So, if you change the dictionary, you must change size_in_bits to match.
    border_bits = 1  # This gives the number of pixels in the margin around the tag code. i.e. black border.

    # gap between markers
    gap_between_markers_in_bits = 2

    # This is the number of markers on the board.
    markers_w = 5  # Number of markers in the X direction.
    markers_h = 7  # Number of markers in the y direction.

    # This gives the physical units.
    # So when you calibrate, you have the same units.
    marker_length = 30  # length of aruco marker (millimetres).

    pixels_per_bit = 10

    generate_aruco_board_for_printing(aruco_dict=ar,
                                      size_in_bits=size_in_bits,
                                      border_bits=border_bits,
                                      gap_between_markers_in_bits=gap_between_markers_in_bits,
                                      marker_length=marker_length,
                                      markers_w=markers_w,
                                      markers_h=markers_h,
                                      pixels_per_bit=pixels_per_bit
                                      )


if __name__ == '__main__':
    main()
