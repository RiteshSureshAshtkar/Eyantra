#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import time

TEAM_ID = 3614

MAX_FRUITS = 4
MATCH_THRESHOLD = 0.08

# ---------------- ROI SETTINGS ----------------
# Horizontal: from LEFT edge up to this fraction of the image width
ROI_RIGHT_FRACTION = 0.27     # same behaviour as your old ROI_LEFT_FRACTION

# Vertical: from TOP edge down to this fraction of the image height
# 0.5 = only top half of the image
ROI_BOTTOM_FRACTION = 0.5

# HSV thresholds for "bad fruit"
HSV_LOWER = np.array([0, 0, 50])
HSV_UPPER = np.array([180, 50, 155])

# Dustbin filter (in base_link coordinates, same as original)
DUSTBIN_POSITION = np.array([-0.806, 0.010, 0.182])
TRASH_IGNORE_RADIUS = 0.25

# ArUco
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
except Exception:
    try:
        ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    except Exception:
        ARUCO_PARAMS = None
ARUCO_DEPTH_ROI = 25

# Fallback camera intrinsics (from your original script)
FALLBACK_CENTER_X = 642.724365234375
FALLBACK_CENTER_Y = 361.9780578613281
FALLBACK_FX = 915.3003540039062
FALLBACK_FY = 914.0320434570312

BASE_FRAME = 'base_link'
CAM_FRAME_DEFAULT = 'camera_link'


class FruitAndFertTFPublisher(Node):
    def __init__(self):
        super().__init__('fruit_and_fertilizer_tf_publisher')
        self.bridge = CvBridge()

        # image buffers
        self.cv_image = None
        self.depth_image = None
        self.color_frame = CAM_FRAME_DEFAULT
        self.latest_camera_info = None

        # intrinsics (updated from CameraInfo if available)
        self.center_x = FALLBACK_CENTER_X
        self.center_y = FALLBACK_CENTER_Y
        self.fx = FALLBACK_FX
        self.fy = FALLBACK_FY

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # fruit tracking
        self.detected_fruits = {}   # id -> dict(pos, last_seen, visible, frame)
        self.next_id_pool = list(range(1, MAX_FRUITS + 1))
        self.initial_locked = False
        self.initial_allowed_ids = set()

        # ArUco detections
        self.aruco_latest = []      # list of {'id','pos_cam','px'}

        # ---------- PUBLISHERS ----------
        # Annotated RGB image
        self.debug_image_pub = self.create_publisher(
            Image,
            '/camera/fruit_detection/image_raw',
            10
        )
        # Mask (mono8)
        self.mask_image_pub = self.create_publisher(
            Image,
            '/camera/fruit_detection/mask',
            10
        )
        # CameraInfo forwarder for RViz Camera display
        self.fruit_cam_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/fruit_detection/camera_info',
            10
        )

        # ---------- SUBSCRIPTIONS ----------
        # Color: original topics
        self.create_subscription(Image,
                                 '/camera/image_raw',
                                 self.color_cb,
                                 6)
        self.create_subscription(CompressedImage,
                                 '/camera/image_raw/compressed',
                                 self.color_compressed_cb,
                                 6)

        # Color: RealSense-style topics
        self.create_subscription(Image,
                                 '/camera/camera/color/image_raw',
                                 self.color_cb,
                                 6)
        self.create_subscription(CompressedImage,
                                 '/camera/camera/color/image_raw/compressed',
                                 self.color_compressed_cb,
                                 6)

        # Depth: original topic
        self.create_subscription(Image,
                                 '/camera/depth/image_raw',
                                 self.depth_cb,
                                 6)

        # Depth: RealSense-style topic
        self.create_subscription(Image,
                                 '/camera/camera/aligned_depth_to_color/image_raw',
                                 self.depth_cb,
                                 6)

        # CameraInfo (optional)
        self.create_subscription(CameraInfo,
                                 '/camera/camera/color/camera_info',
                                 self.camera_info_cb,
                                 6)

        # Main processing timer
        self.last_process_t = 0.0
        self.create_timer(0.12, self._regular_process)

        self.get_logger().info(
            "Fruit and Fertilizer TF publisher started. "
            "Publishing TFs in base_link and debug images + camera_info for RViz."
        )

    # ---------- ROS Callbacks ----------

    def color_cb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if msg.header.frame_id:
                self.color_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"color_cb error: {e}")

    def color_compressed_cb(self, msg: CompressedImage):
        # Fallback if camera publishes only compressed images
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                self.cv_image = img
            if msg.header.frame_id:
                self.color_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"color_compressed_cb error: {e}")

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_image = np.array(depth)
        except Exception as e:
            self.get_logger().warn(f"depth_cb error: {e}")

    def camera_info_cb(self, msg: CameraInfo):
        try:
            # update intrinsics
            self.latest_camera_info = msg
            K = msg.k  # 3x3 row-major
            self.fx = float(K[0])
            self.fy = float(K[4])
            self.center_x = float(K[2])
            self.center_y = float(K[5])

            # ---- forward CameraInfo to /camera/fruit_detection_camera_info ----
            out = CameraInfo()
            out.header.stamp = self.get_clock().now().to_msg()
            # tie it to the same frame as the color image we use
            out.header.frame_id = self.color_frame or msg.header.frame_id
            out.height = msg.height
            out.width = msg.width
            out.distortion_model = msg.distortion_model
            out.d = list(msg.d)
            out.k = list(msg.k)
            out.r = list(msg.r)
            out.p = list(msg.p)
            out.binning_x = msg.binning_x
            out.binning_y = msg.binning_y
            out.roi = msg.roi

            self.fruit_cam_info_pub.publish(out)

        except Exception as e:
            self.get_logger().warn(f"camera_info_cb error: {e}")

    # ---------- Detection Helpers ----------

    def _depth_roi_mean_meters(self, depth_np, cx, cy, roi_size=15):
        if depth_np is None:
            return None
        h, w = depth_np.shape[:2]
        half = roi_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        roi = depth_np[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        # depth could be float (meters) or uint16 (millimeters)
        if np.issubdtype(roi.dtype, np.floating):
            vals = roi[np.isfinite(roi) & (roi > 0.0)]
            if vals.size == 0:
                return None
            return float(np.mean(vals))
        else:
            vals = roi[roi > 0]
            if vals.size == 0:
                return None
            return float(np.mean(vals)) / 1000.0

    def bad_fruit_detection_pixels(self, rgb):
        """Return fruit candidate centers and the HSV mask."""
        h, w, _ = rgb.shape

        # ----- NEW ROI LIMITS -----
        # Horizontal ROI: from x=0 (left edge) to x=ROI_RIGHT_FRACTION * w
        roi_x0 = 0
        roi_x1 = int(w * ROI_RIGHT_FRACTION)

        # Vertical ROI: from y=0 (top edge) to y=ROI_BOTTOM_FRACTION * h
        roi_y0 = 0
        roi_y1 = int(h * ROI_BOTTOM_FRACTION)

        spatial_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(spatial_mask,
                      (roi_x0, roi_y0),
                      (roi_x1, roi_y1),
                      255,
                      -1)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        final_mask = cv2.bitwise_and(mask, mask, mask=spatial_mask)
        final_mask = cv2.erode(final_mask, None, iterations=2)
        final_mask = cv2.dilate(final_mask, None, iterations=2)

        contours, _ = cv2.findContours(final_mask.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            x, y, wbox, hbox = cv2.boundingRect(cnt)
            cx = x + wbox // 2
            cy = y + hbox // 2
            centers.append((cx, cy, x, y, wbox, hbox))

        return centers, final_mask

    def compute_aruco_detections(self):
        """Detect ArUco markers; return list of {'id','pos_cam','px'}."""
        detections = []
        if self.cv_image is None or self.depth_image is None or ARUCO_PARAMS is None:
            return detections

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray,
                                                  ARUCO_DICT,
                                                  parameters=ARUCO_PARAMS)
        if ids is None or len(ids) == 0:
            return detections

        for i, corner in enumerate(corners):
            c = corner.reshape((4, 2))
            cx = int(np.round(np.mean(c[:, 0])))
            cy = int(np.round(np.mean(c[:, 1])))

            avg_depth_m = self._depth_roi_mean_meters(self.depth_image,
                                                      cx,
                                                      cy,
                                                      roi_size=ARUCO_DEPTH_ROI)
            if avg_depth_m is None or avg_depth_m <= 0.03:
                continue

            # pixel -> camera coordinates (same style as your original math)
            x_cam = avg_depth_m * (cx - self.center_x) / self.fx
            y_cam = avg_depth_m * (cy - self.center_y) / self.fy
            z_cam = avg_depth_m

            pos_cam = np.array([x_cam, y_cam, z_cam])
            detections.append({'id': int(ids[i][0]),
                               'pos_cam': pos_cam,
                               'px': (cx, cy)})
        return detections

    def publish_transform(self, child_name: str, pos: np.ndarray,
                          parent_frame: str = BASE_FRAME):
        """Publish a TF from parent_frame to child_name with translation=pos."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_name
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        try:
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f"publish_transform error for '{child_name}': {e}")

    # ---------- PUBLISH DEBUG IMAGES ----------

    def _publish_debug_images(self, img, mask):
        now = self.get_clock().now().to_msg()

        # Annotated BGR image
        try:
            if img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header.stamp = now
                img_msg.header.frame_id = self.color_frame
                self.debug_image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish annotated image: {e}")

        # Mask (mono8)
        try:
            if mask is not None:
                if len(mask.shape) == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask
                mask_msg = self.bridge.cv2_to_imgmsg(mask_gray, encoding='mono8')
                mask_msg.header.stamp = now
                mask_msg.header.frame_id = self.color_frame
                self.mask_image_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish mask image: {e}")

    # ---------- Main Periodic Processing ----------

    def _regular_process(self):
        # Throttle processing slightly
        tnow = time.time()
        if tnow - self.last_process_t < 0.08:
            return
        self.last_process_t = tnow

        if self.cv_image is None or self.depth_image is None:
            return

        img = self.cv_image.copy()
        depth = self.depth_image.copy()

        # ----- 1. ArUco detection & fertilizer TF -----
        self.aruco_latest = self.compute_aruco_detections()
        if self.aruco_latest:
            # choose the marker closest in z (distance)
            best = None
            best_z = float('inf')
            for det in self.aruco_latest:
                z = float(det['pos_cam'][2])
                if z < best_z:
                    best_z = z
                    best = det

            if best is not None:
                pos = best['pos_cam']
                fert_frame = f"{TEAM_ID}_fertilizer_1"
                self.publish_transform(fert_frame, pos, parent_frame=BASE_FRAME)

                # Draw marker on image for visualization
                cx, cy = best['px']
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(img,
                            f"FERT ID:{best['id']}",
                            (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1)

        # ----- 2. Bad fruit detection -----
        detections_px, mask = self.bad_fruit_detection_pixels(img)
        candidates = []
        for (cX, cY, x, y, wbox, hbox) in detections_px:
            h, w = depth.shape[:2]
            if not (0 <= cY < h and 0 <= cX < w):
                continue

            avg_depth_m = self._depth_roi_mean_meters(depth, cX, cY, roi_size=15)
            if avg_depth_m is None or avg_depth_m <= 0.03:
                continue

            # pixel -> camera coords
            x_cam = avg_depth_m * (cX - self.center_x) / self.fx
            y_cam = avg_depth_m * (cY - self.center_y) / self.fy
            z_cam = avg_depth_m

            pos = np.array([x_cam, y_cam, z_cam])

            # ignore near dustbin (approximate, since no TF)
            if np.linalg.norm(pos - DUSTBIN_POSITION) <= TRASH_IGNORE_RADIUS:
                continue

            candidates.append((pos, (cX, cY, x, y, wbox, hbox)))

            # Draw bounding box + center on 'view' image
            cv2.rectangle(img, (x, y), (x + wbox, hbox + y), (0, 0, 255), 2)
            cv2.circle(img, (cX, cY), 4, (255, 0, 0), -1)

        now = time.time()

        # mark all as not visible by default
        for fid, info in self.detected_fruits.items():
            info['visible'] = False

        # candidate → tracked fruit matching
        for pos, pxinfo in candidates:
            best_id = None
            best_dist = float('inf')
            for fid, info in self.detected_fruits.items():
                if 'pos' in info and info['pos'] is not None:
                    d = np.linalg.norm(pos - info['pos'])
                    if d < best_dist:
                        best_dist = d
                        best_id = fid

            if best_id is not None and best_dist <= MATCH_THRESHOLD:
                # update existing fruit
                self.detected_fruits[best_id]['pos'] = pos
                self.detected_fruits[best_id]['last_seen'] = now
                self.detected_fruits[best_id]['visible'] = True
            else:
                # create new fruit if allowed
                if self.initial_locked:
                    self.get_logger().debug("Initial set locked; ignoring new fruit.")
                else:
                    if len(self.next_id_pool) == 0:
                        self.get_logger().debug("No free ID to assign for new fruit.")
                    else:
                        new_id = self.next_id_pool.pop(0)
                        frame_name = f"{TEAM_ID}_bad_fruit_{new_id}"
                        self.detected_fruits[new_id] = {
                            'pos': pos,
                            'last_seen': now,
                            'visible': True,
                            'frame': frame_name,
                        }
                        self.get_logger().info(
                            f"Assigned fruit ID {new_id} -> frame '{frame_name}', pos {pos}"
                        )

        # lock initial set after first batch
        if (not self.initial_locked) and len(self.detected_fruits) > 0:
            self.initial_locked = True
            self.initial_allowed_ids = set(self.detected_fruits.keys())
            self.get_logger().info(
                f"Initial fruit set locked: IDs {sorted(list(self.initial_allowed_ids))}."
            )

        # publish TFs and clean up stale fruits
        for fid, info in list(self.detected_fruits.items()):
            if 'pos' not in info or info['pos'] is None:
                continue

            pos = info['pos']
            frame = info.get('frame', f"{TEAM_ID}_bad_fruit_{fid}")

            self.publish_transform(frame, pos, parent_frame=BASE_FRAME)

            # Remove if stale
            if now - info.get('last_seen', now) > 10.0:
                del self.detected_fruits[fid]
                if not self.initial_locked:
                    self.next_id_pool.append(fid)
                    self.next_id_pool = sorted(self.next_id_pool)
                self.get_logger().info(f"Removed stale fruit ID {fid} (not seen for >10s).")

        # ----- 3. Publish images for RViz -----
        self._publish_debug_images(img, mask)


def main(args=None):
    rclpy.init(args=args)
    node = FruitAndFertTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down (Ctrl+C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
