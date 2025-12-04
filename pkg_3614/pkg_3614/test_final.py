#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified fruit & fertilizer TF publisher.

- ROI cropping controlled by four fraction variables (left/right/top/bottom).
- Detect ArUco -> publish fertilizer TF for the nearest marker.
- Detect HSV "bad fruit" mask -> for each contour with area >= MIN_FRUIT_AREA_PIXELS,
  compute centroid + depth and publish a TF named "{TEAM_ID}_bad_fruit_<ts>_<idx>".
- No ID pooling, no persistent tracking, no MAX_FRUITS limit.
- Keeps camera_info callback to update intrinsics (fallback values used if unavailable).
- Publishes annotated image and mask for RViz/debugging.
"""

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

# ---------------- ROI SETTINGS (fractions 0..1) ----------------
# Crop rectangle in image pixel coordinates:
# left = ROI_LEFT_FRACTION * width
# right = ROI_RIGHT_FRACTION * width
# top = ROI_TOP_FRACTION * height
# bottom = ROI_BOTTOM_FRACTION * height
ROI_LEFT_FRACTION = 0.0
ROI_RIGHT_FRACTION = 0.27
ROI_TOP_FRACTION = 0.0
ROI_BOTTOM_FRACTION = 0.5

# HSV thresholds for "bad fruit" (adjust if needed)
HSV_LOWER = np.array([0, 0, 50])
HSV_UPPER = np.array([180, 50, 155])

# Minimum contour area (in pixels) to be considered a fruit
# Increase to require larger blobs; decrease to allow smaller fruits.
MIN_FRUIT_AREA_PIXELS = 1200

# ArUco settings
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
except Exception:
    try:
        ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    except Exception:
        ARUCO_PARAMS = None
ARUCO_DEPTH_ROI = 25  # px radius for averaging depth around marker centre

# Fallback intrinsics (kept from original)
FALLBACK_CENTER_X = 642.724365234375
FALLBACK_CENTER_Y = 361.9780578613281
FALLBACK_FX = 915.3003540039062
FALLBACK_FY = 914.0320434570312

BASE_FRAME = 'base_link'
CAM_FRAME_DEFAULT = 'camera_link'


class FruitAndFertTFPublisher(Node):
    def __init__(self):
        super().__init__('fruit_and_fertilizer_tf_publisher_simple')
        self.bridge = CvBridge()

        # image buffers
        self.cv_image = None
        self.depth_image = None
        self.color_frame = CAM_FRAME_DEFAULT
        self.latest_camera_info = None

        # intrinsics (may be updated from incoming CameraInfo)
        self.center_x = FALLBACK_CENTER_X
        self.center_y = FALLBACK_CENTER_Y
        self.fx = FALLBACK_FX
        self.fy = FALLBACK_FY

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---------- PUBLISHERS ----------
        # Annotated RGB image
        self.debug_image_pub = self.create_publisher(Image,
                                                     '/camera/fruit_detection/image_raw',
                                                     10)
        # Mask (mono8)
        self.mask_image_pub = self.create_publisher(Image,
                                                    '/camera/fruit_detection/mask',
                                                    10)

        # ---------- SUBSCRIPTIONS ----------
        # Color images (raw & compressed, both common topic variants)
        self.create_subscription(Image,
                                 '/camera/image_raw',
                                 self.color_cb,
                                 6)
        self.create_subscription(CompressedImage,
                                 '/camera/image_raw/compressed',
                                 self.color_compressed_cb,
                                 6)
        self.create_subscription(Image,
                                 '/camera/camera/color/image_raw',
                                 self.color_cb,
                                 6)
        self.create_subscription(CompressedImage,
                                 '/camera/camera/color/image_raw/compressed',
                                 self.color_compressed_cb,
                                 6)

        # Depth images (two common variants)
        self.create_subscription(Image,
                                 '/camera/depth/image_raw',
                                 self.depth_cb,
                                 6)
        self.create_subscription(Image,
                                 '/camera/camera/aligned_depth_to_color/image_raw',
                                 self.depth_cb,
                                 6)

        # CameraInfo to update intrinsics if available
        self.create_subscription(CameraInfo,
                                 '/camera/camera/color/camera_info',
                                 self.camera_info_cb,
                                 6)

        # Main processing timer
        self.last_process_t = 0.0
        self.create_timer(0.12, self._regular_process)

        self.get_logger().info("Simplified Fruit & Fert TF publisher started.")

    # ---------- ROS Callbacks ----------

    def color_cb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if msg.header.frame_id:
                self.color_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"color_cb error: {e}")

    def color_compressed_cb(self, msg: CompressedImage):
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
            self.latest_camera_info = msg
            K = msg.k
            self.fx = float(K[0])
            self.fy = float(K[4])
            self.center_x = float(K[2])
            self.center_y = float(K[5])
        except Exception as e:
            self.get_logger().warn(f"camera_info_cb error: {e}")

    # ---------- Helpers ----------

    def _depth_roi_mean_meters(self, depth_np, cx, cy, roi_size=15):
        """Average a small square ROI around (cx,cy). Returns meters or None."""
        if depth_np is None:
            return None
        h, w = depth_np.shape[:2]
        half = roi_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        roi = depth_np[y0:y1, x0:x1]
        if roi.size == 0:
            return None

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

    def publish_transform(self, child_name: str, pos: np.ndarray, parent_frame: str = BASE_FRAME):
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

    # ---------- ArUco detection ----------

    def compute_aruco_detections(self):
        """Detect ArUco markers; return list of {'id','pos_cam','px'}."""
        detections = []
        if self.cv_image is None or self.depth_image is None or ARUCO_PARAMS is None:
            return detections

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        if ids is None or len(ids) == 0:
            return detections

        for i, corner in enumerate(corners):
            c = corner.reshape((4, 2))
            cx = int(np.round(np.mean(c[:, 0])))
            cy = int(np.round(np.mean(c[:, 1])))

            avg_depth_m = self._depth_roi_mean_meters(self.depth_image, cx, cy, roi_size=ARUCO_DEPTH_ROI)
            if avg_depth_m is None or avg_depth_m <= 0.03:
                continue

            x_cam = avg_depth_m * (cx - self.center_x) / self.fx
            y_cam = avg_depth_m * (cy - self.center_y) / self.fy
            z_cam = avg_depth_m
            pos_cam = np.array([x_cam, y_cam, z_cam])
            detections.append({'id': int(ids[i][0]), 'pos_cam': pos_cam, 'px': (cx, cy)})
        return detections

    # ---------- Debug image publishing ----------

    def _publish_debug_images(self, img, mask):
        now = self.get_clock().now().to_msg()

        try:
            if img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header.stamp = now
                img_msg.header.frame_id = self.color_frame
                self.debug_image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish annotated image: {e}")

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

    # ---------- Main processing ----------

    def _regular_process(self):
        # throttle
        tnow = time.time()
        if tnow - self.last_process_t < 0.08:
            return
        self.last_process_t = tnow

        if self.cv_image is None or self.depth_image is None:
            return

        img_full = self.cv_image.copy()
        depth = self.depth_image.copy()
        h, w = img_full.shape[:2]

        # compute pixel ROI from fractions and clip
        lx = int(round(np.clip(ROI_LEFT_FRACTION, 0.0, 1.0) * w))
        rx = int(round(np.clip(ROI_RIGHT_FRACTION, 0.0, 1.0) * w))
        ty = int(round(np.clip(ROI_TOP_FRACTION, 0.0, 1.0) * h))
        by = int(round(np.clip(ROI_BOTTOM_FRACTION, 0.0, 1.0) * h))

        # ensure ROI is valid: left < right, top < bottom
        if rx <= lx:
            rx = w
        if by <= ty:
            by = h

        # crop for processing
        roi_img = img_full[ty:by, lx:rx]
        roi_mask_full = np.zeros((h, w), dtype=np.uint8)  # full-size mask for publishing
        roi_mask = None

        # ---------- 1) ArUco detection & fertilizer TF ----------
        arucos = self.compute_aruco_detections()
        if arucos:
            # choose the marker closest in z (distance)
            best = min(arucos, key=lambda d: float(d['pos_cam'][2]))
            pos = best['pos_cam']
            fert_frame = f"{TEAM_ID}_fertilizer_1"
            # publish in BASE_FRAME (keeps behavior compatible with original)
            self.publish_transform(fert_frame, pos, parent_frame=BASE_FRAME)
            # draw on full image
            cx, cy = best['px']
            cv2.circle(img_full, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(img_full, f"FERT ID:{best['id']}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---------- 2) Bad fruit detection via HSV mask ----------
        if roi_img is not None and roi_img.size > 0:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours in roi-coordinates, then convert to full-image coords
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fruit_count = 0
            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < MIN_FRUIT_AREA_PIXELS:
                    continue  # too small -> ignore

                x, y, wbox, hbox = cv2.boundingRect(cnt)
                cX = lx + x + wbox // 2
                cY = ty + y + hbox // 2

                # get average depth around centroid
                avg_depth_m = self._depth_roi_mean_meters(depth, cX, cY, roi_size=15)
                if avg_depth_m is None or avg_depth_m <= 0.03:
                    continue

                # convert to camera coordinates (same math as original)
                x_cam = avg_depth_m * (cX - self.center_x) / self.fx
                y_cam = avg_depth_m * (cY - self.center_y) / self.fy
                z_cam = avg_depth_m
                pos = np.array([x_cam, y_cam, z_cam])

                # publish TF for this fruit (timestamped name, no persistent IDs)
                ts = int(time.time() * 1000)
                child_name = f"{TEAM_ID}_bad_fruit_{ts}_{idx}"
                self.publish_transform(child_name, pos, parent_frame=BASE_FRAME)
                fruit_count += 1

                # draw on full image for visualization
                cv2.rectangle(img_full, (lx + x, ty + y), (lx + x + wbox, ty + y + hbox), (0, 0, 255), 2)
                cv2.circle(img_full, (cX, cY), 4, (255, 0, 0), -1)
                cv2.putText(img_full, f"Fruit A:{int(area)}", (cX + 6, cY - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # put roi mask into full-size mask for publishing
            roi_mask_full[ty:by, lx:rx] = mask
            roi_mask = roi_mask_full

            cv2.rectangle(img_full, (lx, ty), (rx, by), (255, 255, 0), 1)
            cv2.putText(img_full, f"ROI fruits:{fruit_count}", (lx + 4, ty + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        else:
            roi_mask = np.zeros((h, w), dtype=np.uint8)

        # publish annotated image + mask
        self._publish_debug_images(img_full, roi_mask)

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
