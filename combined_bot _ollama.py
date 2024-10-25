import json
import numpy as np
import threading
import queue
import cv2
from sic_framework.devices.desktop import Desktop
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, Dialogflow)
from sic_framework.core import utils_cv2
from sic_framework.core.message_python2 import BoundingBoxesMessage
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf
from sic_framework.services.face_detection.face_detection import FaceDetection
import requests

response_event = threading.Event()
local_ai_response = None

def generate_response(prompt):
    url = "http://127.0.0.1:11434/api/chat"
    body = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "system",
                "content": "you are a robot assistant"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=body)

    if response.status_code == 200:
        data = response.json()
        return data['message']['content']
    else:
        print(f"Error: {response.status_code}")
        return None

def on_dialog(message):
    global local_ai_response
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)
            local_ai_response = generate_response(message.response.recognition_result.transcript)
            print("Local AI:", local_ai_response)
            response_event.set()
            
def audio_thread_function():
    desktop_audio = Desktop()
    keyfile_json = json.load(open("robot_key.json"))
    conf = DialogflowConf(keyfile_json=keyfile_json, sample_rate_hertz=44100, language="en")
    dialogflow = Dialogflow(ip='localhost', conf=conf)
    dialogflow.connect(desktop_audio.mic)
    dialogflow.register_callback(on_dialog)
    
    print("Audio system ready.")
    x = np.random.randint(10000)

    try:
        for i in range(25):
            print(" ----- Conversation turn", i)
            contexts_dict = {"name": 1}
            reply = dialogflow.request(GetIntentRequest(x, contexts_dict))
            print("The detected intent:", reply.intent)

            if reply.fulfillment_message:
                text = reply.fulfillment_message
                print("Reply:", text)
            
            response_event.wait()  
            response_event.clear() 

    except KeyboardInterrupt:
        print("Stopping audio system.")
        dialogflow.stop()

imgs_buffer = queue.Queue(maxsize=1)
faces_buffer = queue.Queue(maxsize=1)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

def on_faces(message: BoundingBoxesMessage):
    faces_buffer.put(message.bboxes)

def video_thread_function():
    conf = DesktopCameraConf(fx=1.0, fy=1.0, flip=-1)
    desktop_video = Desktop(camera_conf=conf)
    face_rec = FaceDetection()
    face_rec.connect(desktop_video.camera)
    desktop_video.camera.register_callback(on_image)
    face_rec.register_callback(on_faces)

    print("Video system ready.")

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

if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_thread_function)
    video_thread = threading.Thread(target=video_thread_function)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()
