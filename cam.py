import cv2
import argparse
from stitcher import CVStitcher, Stitcher
from dewarp import fishEyeUnwarp
import time

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", action='store_true',
                help="Show debug statements.")
ap.add_argument("-s", "--stitch", action='store_true',
                help="Stitch external camera feeds.")
ap.add_argument("--width", nargs='?', default=1920, type=int)
ap.add_argument("--height", nargs='?', default=1080, type=int)
args = ap.parse_args()


class cameraSystem:

    def __init__(self, debug, stitch, res_width, res_height):
        print("Initializing cameras...")
        self.debug = debug
        self.stitch = stitch

        # TODO: Check why one ELP camera not able to use DSHOW?
        self.cabinCam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.extCam0 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        self.extCam1 = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
        self.extCam2 = cv2.VideoCapture(3 + cv2.CAP_DSHOW)
        self.extCam3 = cv2.VideoCapture(4)
        self.extCams = [self.extCam0, self.extCam1, self.extCam2, self.extCam3]

        self.extFrames = []
        self.pano = None
        self.cabinFrame = None
        self.f = fishEyeUnwarp(args.width, args.height,
                               args.width, args.height, 180, 125)

        for cam in self.extCams:
            cam.set(self.capPropId("FOURCC"), cv2.VideoWriter_fourcc(*'MJPG'))
            cam.set(self.capPropId("FRAME_WIDTH"), args.width)
            cam.set(self.capPropId("FRAME_HEIGHT"), args.height)
            cam.set(self.capPropId("FPS"), 20)
        print("Initialization complete.")

    def decode_fourcc(self, cc):
        return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

    def capPropId(self, prop):
        return getattr(cv2, "CAP_PROP_" + prop)

    # TODO: Process internal cabin camera.
    def process(self):
        while(True):
            start_time = time.time()
            # Capture frame-by-frame
            self.extFrames.clear()
            for cam in self.extCams:
                ret, frame = cam.read()
                frame = self.f.unwarp(frame)
                self.extFrames.append(frame)
            # Our operations on the frame come here
            # Display the resulting frame
            if self.stitch:
                # s = CVStitcher(args.debug)
                s = Stitcher(self.debug)
                pano = s.stitch(self.extFrames)
                # cv2.resize(pano, (320, 180), interpolation=cv2.INTER_AREA)
                cv2.imshow("pano", pano)
            if self.debug:
                for i in range(len(self.extCams)):
                    cv2.imshow('frame' + str(i),
                               cv2.resize(self.extFrames[i], (480, 270),
                                          interpolation=cv2.INTER_AREA))
                for cam in self.extCams:
                    print(self.decode_fourcc(cam.get(cv2.CAP_PROP_FOURCC)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps = 1.0 / (time.time() - start_time)
            print("[INFO] approx. FPS: {:.2f}".format(fps))

        # Cleanly exit.
        for cam in self.extCams:
            cam.release()
        cv2.destroyAllWindows()


def main():
    c = cameraSystem(args.debug, args.stitch, args.width, args.height)
    c.process()


main()
