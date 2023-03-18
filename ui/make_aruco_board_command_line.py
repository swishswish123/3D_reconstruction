# -*- coding: utf-8 -*-

""" Command line processing for making ArUco board. """

import argparse
import cv2
from aruco.make_aruco_board import make_aruco_board_image


def create_aruco_parser():
    """
    Creates the command line parser.
    :return: argparse.ArgumentParser()
    """
    parser = argparse.ArgumentParser(description='make_aruco_board')

    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Output file name")

    parser.add_argument("-x", "--width",
                        type=int,
                        default=5,
                        help="Number of tags wide (x direction).")

    parser.add_argument("-y", "--height",
                        type=int,
                        default=7,
                        help="Number of tags high (y direction).")

    parser.add_argument("-s", "--size",
                        type=int,
                        default=30,
                        help="Size of tag in millimetres.")

    parser.add_argument("-p", "--pixels",
                        type=int,
                        default=10,
                        help="Number of pixels for each line of the ArUco pattern.")

    return parser


def main(args=None):
    """
    Entry point for the make_aruco_board program.
    """

    # Creates command line parser.
    parser = create_aruco_parser()
    parsed_args = parser.parse_args(args)

    # This is a separate function, with no I/O, so its easy
    # to unit test without writing junk files.
    image = make_aruco_board_image(parsed_args.width,
                                   parsed_args.height,
                                   parsed_args.size,
                                   parsed_args.pixels
                                   )

    # Should really validate that the output file name ends with .png or .jpg
    # or else OpenCV won't know what file format to write.
    # Leave that for now as an 'exercise for the reader'!
    cv2.imwrite(parsed_args.output, image)
