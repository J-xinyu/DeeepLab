#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import os


# ======================
# Camera intrinsics
# ======================
camera_matrix = np.array([
    [599.001467, 0.0,     319.613991],
    [0.0,        597.151197, 241.176052],
    [0,          0,        1]
], dtype=np.float32)

dist_coeffs = np.array(
    [0.109197, -0.201992, -0.004476, -0.004232, 0.0],
    dtype=np.float32
)

# ======================
# Lidar→Camera extrinsics
# ======================
rvec_base = np.array([1.206934, -1.1325423, 1.29024], np.float32)
tvec_base = np.array([-0.74037988, 0.3872010, -1.10953436], np.float32)


# ============================================================
#   PART 1: Curve + Surface Fitting (from CSV)
# ============================================================
class Fitter:
    def __init__(self):
        self.curve_pub = rospy.Publisher("/spline_marker", Marker, queue_size=10)
        self.surface_pub = rospy.Publisher("/spline_surface", Marker, queue_size=10)

    def load_points(self, file):
        try:
            pts = np.loadtxt(file, delimiter=',')
            rospy.loginfo(f"Loaded CSV: {pts.shape}")
            return pts
        except Exception as e:
            rospy.logerr(f"Failed to load CSV: {e}")
            return None

    def close_curve(self, pts):
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        return pts

    def spline_fit(self, pts):
        tck, u = splprep(pts.T, s=0, per=1)
        u_new = np.linspace(0, 1, len(pts) * 200)  
        spline = np.array(splev(u_new, tck)).T
        return spline

    def create_curve_marker(self, pts):
        mk = Marker()
        mk.header.frame_id = "map"
        mk.type = Marker.LINE_STRIP
        mk.scale.x = 0.01
        mk.color.r = 0
        mk.color.g = 1
        mk.color.b = 0
        mk.color.a = 1

        for x, y, z in pts:
            mk.points.append(Point(x, y, z))
        return mk

    def create_surface_marker(self, curve):
        mk = Marker()
        mk.header.frame_id = "map"
        mk.type = Marker.TRIANGLE_LIST
        mk.scale.x = mk.scale.y = mk.scale.z = 1.0

        mk.color.r = 0.2
        mk.color.g = 0.9
        mk.color.b = 0.9
        mk.color.a = 0.6

        P0 = curve[0]
        for i in range(1, len(curve) - 1):
            tri = [P0, curve[i], curve[i + 1]]
            for x, y, z in tri:
                mk.points.append(Point(x, y, z))
        return mk

    def run(self, csv_file):
        pts = self.load_points(csv_file)
        if pts is None:
            return

        pts = self.close_curve(pts)
        spline_pts = self.spline_fit(pts)

        curve_mk = self.create_curve_marker(spline_pts)
        surf_mk = self.create_surface_marker(spline_pts)

        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            t = rospy.Time.now()
            curve_mk.header.stamp = t
            surf_mk.header.stamp = t
            self.curve_pub.publish(curve_mk)
            self.surface_pub.publish(surf_mk)
            rate.sleep()


# ============================================================
#   PART 2: Projection (image + pointcloud + fitted curve/surface)
# ============================================================
class Projector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.cloud = None
        self.curve_pts = []
        self.surface_tris = []

        # Define the range for filtering the point cloud
        self.x_range = [0, 100]  # [min, max]
        self.y_range = [-12, 12]
        self.z_range = [-1.5, 5]

        rospy.Subscriber("/camera/color/image_bright", Image, self.cb_image)
        rospy.Subscriber("/frame_detection_Viz", PointCloud2, self.cb_cloud)
        rospy.Subscriber("/spline_marker", Marker, self.cb_curve)
        rospy.Subscriber("/spline_surface", Marker, self.cb_surface)

        cv2.namedWindow("Projection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Adjust")
        self.init_trackbars()

        rospy.Timer(rospy.Duration(0.03), self.refresh)

    def init_trackbars(self):
        cv2.createTrackbar("r_x", "Adjust", 100, 200, lambda x: None)
        cv2.createTrackbar("r_y", "Adjust", 100, 200, lambda x: None)
        cv2.createTrackbar("r_z", "Adjust", 100, 200, lambda x: None)
        cv2.createTrackbar("t_x", "Adjust", 100, 400, lambda x: None)
        cv2.createTrackbar("t_y", "Adjust", 100, 400, lambda x: None)
        cv2.createTrackbar("t_z", "Adjust", 100, 400, lambda x: None)

        cv2.createTrackbar("surf_on", "Adjust", 1, 1, lambda x: None)
        cv2.createTrackbar("surf_a", "Adjust", 50, 100, lambda x: None)

    def extrinsic(self):
        r = rvec_base + (np.array([
            cv2.getTrackbarPos("r_x", "Adjust") - 100,
            cv2.getTrackbarPos("r_y", "Adjust") - 100,
            cv2.getTrackbarPos("r_z", "Adjust") - 100
        ]) / 100.0)

        t = tvec_base + (np.array([
            cv2.getTrackbarPos("t_x", "Adjust") - 100,
            cv2.getTrackbarPos("t_y", "Adjust") - 100,
            cv2.getTrackbarPos("t_z", "Adjust") - 100
        ]) / 100.0)

        return r.astype(np.float32), t.astype(np.float32)

    # ---- Callbacks ----
    def cb_image(self, msg): self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def cb_cloud(self, msg): self.cloud = msg
    def cb_curve(self, msg):
        self.curve_pts = [(p.x, p.y, p.z) for p in msg.points]

    def cb_surface(self, msg):
        self.surface_tris = []
        pts = msg.points
        for i in range(0, len(pts), 3):
            tri = [(pts[i].x, pts[i].y, pts[i].z),
                   (pts[i + 1].x, pts[i + 1].y, pts[i + 1].z),
                   (pts[i + 2].x, pts[i + 2].y, pts[i + 2].z)]
            self.surface_tris.append(tri)

    # ---- Rendering ----
    def refresh(self, evt):
        if self.image is None:
            return

        img = self.image.copy()
        rvec, tvec = self.extrinsic()

        # D) Curve (绿色)
        if len(self.curve_pts) > 1:
            arr = np.array(self.curve_pts, np.float32)
            proj, _ = cv2.projectPoints(arr, rvec, tvec, camera_matrix, dist_coeffs)
            proj = proj.reshape(-1, 2).astype(int)
            for i in range(len(proj) - 1):
                cv2.line(img, tuple(proj[i]), tuple(proj[i + 1]), (0, 255, 0), 2)  # 将线条宽度改为1，绿色

        # E) Surface fill
        surf_on = cv2.getTrackbarPos("surf_on", "Adjust")
        alpha = cv2.getTrackbarPos("surf_a", "Adjust") / 100.0

        if surf_on == 1 and len(self.surface_tris) > 0:
            overlay = img.copy()
            for tri in self.surface_tris:
                tri_np = np.array(tri, np.float32)
                p2, _ = cv2.projectPoints(tri_np, rvec, tvec, camera_matrix, dist_coeffs)
                pts = p2.reshape(-1, 2).astype(int)
                cv2.fillConvexPoly(overlay, pts, (255, 220, 180))  
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # F) Point cloud 投影（橙黄色）- 过滤后的点云
        if self.cloud is not None:
            xyz = []
            for p in pc2.read_points(self.cloud, field_names=("x", "y", "z"), skip_nans=True):
                # 过滤点云：仅显示在指定范围内的点
                if (self.x_range[0] <= p[0] <= self.x_range[1] and
                    self.y_range[0] <= p[1] <= self.y_range[1] and
                    self.z_range[0] <= p[2] <= self.z_range[1]):
                    xyz.append([p[0], p[1], p[2]])

            if xyz:
                arr = np.array(xyz, np.float32)
                proj, _ = cv2.projectPoints(arr, rvec, tvec, camera_matrix, dist_coeffs)
                for p in proj:
                    x, y = int(p[0][0]), int(p[0][1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(img, (x, y), 1, (0, 165, 255), -1)  

        cv2.imshow("Projection", img)
        cv2.waitKey(1)


# ============================================================
#   MAIN
# ============================================================
if __name__ == "__main__":
    rospy.init_node("final_fitting_projector", anonymous=True)

    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "points.csv")

    fitter = Fitter()
    projector = Projector()

    # Fitting runs in background (publishers)
    import threading
    threading.Thread(target=fitter.run, args=(csv,), daemon=True).start()

    rospy.spin()

