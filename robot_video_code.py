import queue
import cv2
from sic_framework.core import utils_cv2
from sic_framework.core.message_python2 import BoundingBoxesMessage
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf
from sic_framework.devices.desktop import Desktop
from sic_framework.services.face_detection.face_detection import FaceDetection

imgs_buffer = queue.Queue(maxsize=1)
faces_buffer = queue.Queue(maxsize=1)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

def on_faces(message: BoundingBoxesMessage):
    faces_buffer.put(message.bboxes)

conf = DesktopCameraConf(fx=1.0, fy=1.0, flip=-1)
desktop = Desktop(camera_conf=conf)
face_rec = FaceDetection()
face_rec.connect(desktop.camera)
desktop.camera.register_callback(on_image)
face_rec.register_callback(on_faces)

while True:
    img = imgs_buffer.get()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = faces_buffer.get()
    if faces:
        for face in faces:
            utils_cv2.draw_bbox_on_image(face, img_rgb)

    cv2.imshow('Face Detection', img_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
