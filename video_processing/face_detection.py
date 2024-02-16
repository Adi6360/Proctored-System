import cv2
import mediapipe as mp

class Face:
    def __init__(self):
        self.bbox = None
        self.img = None
        self.name = None
        self.distance = None
        self.confidence = None
        self.landmarks = None
        self.mouth = None
        self.head = None
        self.eye = None
        self.spoof = None
        self.spoof_score = None

# Crop face based on its bounding box
def get_face(frame, bbox):
    real_h, real_w, c = frame.shape
    x, y, w, h = bbox
    y1 = max(0, y)
    x1 = max(0, x)
    y2 = min(real_h, y + h)
    x2 = min(real_w, x + w)
    face = frame[y1:y2, x1:x2, :]
    return face

def detect_faces(frame, confidence=0.7):
    faces = None
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=confidence) as face_detector:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(frame_rgb)

    if results.detections:
        faces = []
        for _, detection in enumerate(results.detections):
            face = Face()
            bbox = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face.bbox = bbox
            face.confidence = detection.score
            face.img = get_face(frame, face.bbox)
            faces.append(face)

    return faces


