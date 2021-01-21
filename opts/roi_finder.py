import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input image to ROI image")
    parser.add_argument("--image-path", type=str, help="Absolute path where Input image exist")
    parser.add_argument("--output-path", type=str, help="Absolute path where Output image saved", default="./output")