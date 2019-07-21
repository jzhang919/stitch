import numpy as np
import cv2


class fishEyeUnwarp:
    def __init__(self, Ws, Hs, Wd, Hd, hfovd=180.0, vfovd=180.0):
        print("Initializing dewarp matrix...")
        self.x_map, self.y_map = self.buildMap(Ws, Hs, Wd, Hd, hfovd, vfovd)

    def buildMap(self, Ws, Hs, Wd, Hd, hfovd, vfovd):
        # Build the fisheye mapping
        map_x = np.zeros((Hd, Wd), np.float32)
        map_y = np.zeros((Hd, Wd), np.float32)
        vfov = (vfovd / 180.0) * np.pi
        hfov = (hfovd / 180.0) * np.pi
        vstart = ((180.0 - vfovd) / 180.00) * np.pi / 2.0
        hstart = ((180.0 - hfovd) / 180.00) * np.pi / 2.0
        count = 0
        # need to scale to changed range from our
        # smaller cirlce traced by the fov
        xmax = np.sin(np.pi / 2.0) * np.cos(vstart)
        xmin = np.sin(np.pi / 2.0) * np.cos(vstart + vfov)
        xscale = xmax - xmin
        xoff = xscale / 2.0
        zmax = np.cos(hstart)
        zmin = np.cos(hfov + hstart)
        zscale = zmax - zmin
        zoff = zscale / 2.0
        # Fill in the map, this is slow but
        # we could probably speed it up
        # since we only calc it once, whatever
        for y in range(0, int(Hd)):
            for x in range(0, int(Wd)):
                count = count + 1
                phi = vstart + (vfov * ((float(x) / float(Wd))))
                theta = hstart + (hfov * ((float(y) / float(Hd))))
                xp = ((np.sin(theta) * np.cos(phi)) + xoff) / zscale
                zp = ((np.cos(theta)) + zoff) / zscale
                xS = Ws - (xp * Ws)
                yS = Hs - (zp * Hs)
                map_x.itemset((y, x), int(xS))
                map_y.itemset((y, x), int(yS))

        return map_x, map_y

    def unwarp(self, img):
        # apply the unwarping map to our image
        output = cv2.remap(img, self.x_map, self.y_map, cv2.INTER_LINEAR)
        return output
