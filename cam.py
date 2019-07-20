import cv2
import argparse
from stitcher import Stitch

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", action='store_true',
                help="Show debug statements.")
args = ap.parse_args()


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def capPropId(prop):
    return getattr(cv2, "CAP_PROP_" + prop)


def main():
    # Intel Realsense Camera needs CAP_DSHOW
    print("Initializing cameras...")
    cam0 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cam1 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    cam2 = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
    cam3 = cv2.VideoCapture(3 + cv2.CAP_DSHOW)
    cam4 = cv2.VideoCapture(4)
    cams = [cam1, cam2]
    frames = []
    for cam in cams:
        cam.set(capPropId("FOURCC"), cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(capPropId("FRAME_WIDTH"), 1920)
        cam.set(capPropId("FRAME_HEIGHT"), 1080)
        cam.set(capPropId("FPS"), 15)
    print("Initialization complete.")

    while(True):
        # Capture frame-by-frame
        frames.clear()
        for cam in cams:
            ret, frame = cam.read()
            frames.append(frame)
        # Our operations on the frame come here
        # Display the resulting frame
        stitcher = Stitch(frames)
        stitcher.leftshift()
        stitcher.rightshift()
        cv2.imshow("Stitched", stitcher.leftImage)
        if args.debug:
            for i in range(len(cams)):
                cv2.imshow('frame' + str(i),
                           cv2.resize(frames[i], (320, 180),
                                      interpolation=cv2.INTER_AREA))
            for cam in cams:
                print(decode_fourcc(cam.get(cv2.CAP_PROP_FOURCC)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()


main()
