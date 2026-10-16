"""Find the phone camera's focal length and lens distortion from a ChArUco video."""
import sys
import cv2
import numpy as np

video, square_mm = sys.argv[1], float(sys.argv[2])
square = square_mm / 1000.0
marker = square * 26.0 / 35.0           # same ratio as the printed board

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((5, 7), square, marker, dictionary)
detector = cv2.aruco.CharucoDetector(board)

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)          # give us the frames as the sensor saw them
angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

obj_pts, img_pts, size, i = [], [], None, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame.shape[0] > frame.shape[1]:            # still upright: undo OpenCV's rotation
        undo = cv2.ROTATE_90_COUNTERCLOCKWISE if angle == 90 else cv2.ROTATE_90_CLOCKWISE
        frame = cv2.rotate(frame, undo)
    i += 1
    if i % 10:                                     # use every 10th frame
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 10:
        continue
    o, p = board.matchImagePoints(corners, ids)
    obj_pts.append(o)
    img_pts.append(p)

print(f"rotation label: {angle}   frames used: {len(obj_pts)}   image size: {size}")
rms, K, dist, _, _ = cv2.calibrateCamera(
    obj_pts, img_pts, size, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
print(f"reprojection error: {rms:.3f} px")
print("camera matrix:\n", K.round(1))
np.savez("data/camera_calib.npz", K=K, dist=dist, size=size, rms=rms)
print("saved data/camera_calib.npz")