import numpy as np
import cv2


def reorder(polygons):
    polygons = polygons.reshape((4,2))
    p_new = np.zeros((4,2))
    diff = np.diff(polygons, axis=1)
    add = np.sum(polygons, axis=1)
    p_new[3] = polygons[np.argmax(add)]
    p_new[0] = polygons[np.argmin(add)]
    
    p_new[1] = polygons[np.argmin(diff)]
    p_new[2] = polygons[np.argmax(diff)]
    
    return p_new.astype("float32")

def bestPolygon(contour):
    for idx in range(len(contour)):
        cnt = contour[idx]
        l = cv2.arcLength(contour[idx], True)
        epsilon = 0.1 * l
        poly = cv2.approxPolyDP(cnt, epsilon, True)
        if len(poly) == 4:
            return poly.reshape((4,2))
    return None 

def scan(frame):
	data_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	data_blurred = cv2.GaussianBlur(data_gray, (17,17), 17/6)
	thresh, threshold = cv2.threshold(data_blurred, 0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	data_contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = sorted(data_contours, reverse=True, key=cv2.contourArea)
	polygons = bestPolygon(contours)
	return polygons

def transform(polygons, frame, dst=None):
	if dst is None:
		dst = np.float32([[0,0],[768, 0],[0,1024],[768, 1024]])
	src = reorder(polygons)
	M = cv2.getPerspectiveTransform(src, dst)
	target_im = cv2.warpPerspective(frame, M, (768, 1024))

	return target_im

cam = cv2.VideoCapture("videoroot")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
source_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
writer = cv2.VideoWriter('output.mp4', fourcc, 10.0, (source_w, source_h))
while True:
	ret, frame = cam.read()
	if ret:

		polygons = scan(frame)
		key = cv2.waitKey(1)
		if key & 0xFF == ord("q"):
			break
		if key & 0xFF == ord("s"):
			target_im = transform(polygons, frame)
			target_im_gray = cv2.cvtColor(target_im, cv2.COLOR_BGR2GRAY)
			adjusted_im = np.clip((target_im_gray/255 - 0.5)*9500+ 0.5 + 127, 0, 255).astype("uint")
			target_im = cv2.cvtColor(target_im_gray, cv2.COLOR_GRAY2BGR)
			break
		if polygons is not None:
			for p in polygons:
				cv2.circle(frame, tuple(p), 10, (0, 204, 255), -1)
			polygons = reorder(polygons)
			cv2.line(frame, tuple(polygons[0]), tuple(polygons[1]), (0,204,255), 3)
			cv2.line(frame, tuple(polygons[0]), tuple(polygons[2]), (0,204,255), 3)
			cv2.line(frame, tuple(polygons[1]), tuple(polygons[3]), (0,204,255), 3)
			cv2.line(frame, tuple(polygons[2]), tuple(polygons[3]), (0,204,255), 3)

		cv2.imshow("Scanner", frame)
		writer.write(frame)
	else:
		break
cv2.imshow("Scanner", target_im)
target_im = cv2.resize(target_im, (source_w, source_h))
for i in range(30):
	writer.write(target_im)
cv2.waitKey(0)
writer.release()
cam.release()
cv2.destroyAllWindows()