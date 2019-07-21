import cv2
import argparse
from stitcher import CVStitcher, Stitcher
from dewarp import fishEyeUnwarp

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", action='store_true',
                help="Show debug statements.")
ap.add_argument("-s", "--stitch", action='store_true',
                help="Stitch external camera feeds.")
ap.add_argument("--width", nargs='?', default=1920, type=int)
ap.add_argument("--height", nargs='?', default=1080, type=int)
args = ap.parse_args()


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


defq capPropId(prop):
    return getattr(cv2, "CAP_PROP_" + prop)


def main():
    # Intel Realsense Camera needs CAP_DSHOW
    print("Initializing cameras...")
    cam0 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cam1 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    cam2 = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
    cam3 = cv2.VideoCapture(3 + cv2.CAP_DSHOW)
    cam4 = cv2.VideoCapture(4)
    cams = [cam1, cam2, cam3, cam4]
    # f = fishEyeUnwarp(args.width, args.height, args.width, args.height, 180, 125)
    frames = []
    for cam in cams:
        cam.set(capPropId("FOURCC"), cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(capPropId("FRAME_WIDTH"), args.width)
        cam.set(capPropId("FRAME_HEIGHT"), args.height)
        cam.set(capPropId("FPS"), 15)
    print("Initialization complete.")

    while(True):
        # Capture frame-by-frame
        frames.clear()
        for cam in cams:
            ret, frame = cam.read()
            # frame = f.unwarp(frame)
            frames.append(frame)
        # Our operations on the frame come here
        # Display the resulting frame
        if args.stitch:
            # s = CVStitcher(args.debug)
            s = Stitcher(args.debug)
            pano = s.stitch(frames)
            # cv2.resize(pano, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.imshow("pano", pano)
        if args.debug:
            for i in range(len(cams)):
                cv2.imshow('frame' + str(i),
                           cv2.resize(frames[i], (480, 270),
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
