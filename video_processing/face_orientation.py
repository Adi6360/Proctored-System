import cv2
import numpy as np
import math

font = cv2.FONT_HERSHEY_SIMPLEX

class PCF:
    def __init__(
            self,
            near=1,
            far=10000,
            frame_height=1920,
            frame_width=1080,
            fy=1074.520446598223,
    ):
        self.near = near
        self.far = far
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fy = fy

        fov_y = 2 * np.arctan(frame_height / (2 * fy))
        height_at_near = 2 * near * np.tan(0.5 * fov_y)
        width_at_near = frame_width * height_at_near / frame_height

        self.fov_y = fov_y
        self.left = -0.5 * width_at_near
        self.right = 0.5 * width_at_near
        self.bottom = -0.5 * height_at_near
        self.top = 0.5 * height_at_near

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 0
    rear_depth = 0
    front_size = 15
    front_depth = 15
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(rotation_vector, translation_vector, camera_matrix, val)
    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)

def get_2d_points(rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix, dir="horiz"):
    rear_size = 0
    rear_depth = 0
    front_size = 15
    front_depth = 15
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(rotation_vector, translation_vector, camera_matrix, val)
    x, y = None, None
    if dir == "horiz":
        y = tuple((point_2d[5] + point_2d[8]) // 2)
        x = tuple(point_2d[2])
    elif dir == "vert":
        y = tuple((point_2d[5] + point_2d[6]) // 2)
        x = tuple(point_2d[2])
    return x, y

def headpose_est(frame, faces, hland):
    landmarks = hland[:468, :].T

    points_idx = [4, 33, 61, 199, 263, 291]
    frame_height, frame_width, channels = frame.shape
    # pseudo camera internals
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))
    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    def get_metric_landmarks(landmarks, pcf):
        # Implement the functionality of get_metric_landmarks here
        pass

    metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None, :])
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    p1, p2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, dir="vert")
    x1, x2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, dir="horiz")
    draw_annotation_box(frame, rotation_vector, translation_vector, camera_matrix)

    # Calculate angles
    ang1 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    ang2 = math.degrees(math.atan2(x2[1] - x1[1], x2[0] - x1[0]))

    tempstr = "Head Straight"
    if ang2 >= 20:
        tempstr = "Head left"
        cv2.putText(frame, 'Head left', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif ang2 <= -20:
        tempstr = "Head right"
        cv2.putText(frame, 'Head right', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif ang1 >= 20:
        tempstr = "Head up"
        cv2.putText(frame, 'Head up', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif ang1 <= -20:  # 10
        tempstr = "Head down"
        cv2.putText(frame, 'Head down', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        tempstr = "Head Straight"
        cv2.putText(frame, 'Head straight', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return tempstr
