class Homography:
	def __init__(self,spatial_points, rvec, tvecs):
		self.r =

def calibrate(spatial_points, pixel_points, img_dim):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([spatial_points], [pixel_points], img_dim,None,None)