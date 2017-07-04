import numpy as np
import cv2
import sys
import time
from PIL import Image


start_x,start_y = -1,-1
end_x, end_y = -1,-1


img = np.zeros((1920,1080,3), np.uint8)
img_h,img_w,img_channel=img.shape

#bg = Image.open("./callout-bubble-2.png")
#bg_w, bg_h = bg.size

def mouseeventproc(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        #cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),-1)


def main():
    # opencv related content
    cv2.namedWindow('view')
    cv2.startWindowThread()
    cv2.setMouseCallback('view', mouseeventproc)
    cap = cv2.VideoCapture(2)
    #ret = cap.set(3,1280);
    #ret = cap.set(4,720);
    #ret = cap.set(3,960);
    #ret = cap.set(4,720);
    ret = cap.set(3,1920);
    ret = cap.set(4,1080);

    global img, start_x, start_y, end_x, end_y, gap_height

    while(True):
        ret, img = cap.read()
        #img = cv2.flip(img, 1)
        H, W = img.shape[:2]
        keycode = cv2.waitKey(1) & 0xff


        if keycode == ord('q'):
            break;
        elif keycode == ord('c'):
            # generate callout styple image
            crop = img[start_y:end_y, start_x:end_x]
            h,w,c = crop.shape
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.rectangle(crop, (1, 1), (w-1,h-1), (0,0,255), 5)
            cv2.imwrite(timestr + ".png",crop)
            fy = float(h)/H
            fx = float(w)/W

            img2 = cv2.resize(img, None,fx=fx, fy=fx, interpolation = cv2.INTER_CUBIC)
            h1,w1,c1 = img2.shape
            gap_height = int(h * 0.5)
            height = h1 + h + gap_height
            width = max(w1,w)
            output = np.zeros((height,width,3))

            y = 0;
            output[y:y+h, 0:w] = crop
            y = y + h + gap_height
            output[y:y+h1, 0:w1] = img2

            ix_1 = int(start_x * fx)
            iy_1 = int(start_y * fx)
            width_1 = int((end_x - start_x) * fx)
            cv2.arrowedLine(output, (ix_1, iy_1 + y), (0, h), (0,0,255), 2)
            cv2.arrowedLine(output, (ix_1 + width_1, iy_1 + y), (w, h), (0,0,255), 2)
            cv2.imwrite(timestr+"_callout"+".png", output)

        cv2.imshow('view', img)

    cv2.destroyWindow('view')
    cap.release()

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        #controller.remove_listener(listener)
        pass

if __name__ == "__main__":
    main()
