import json
import numpy as np
import multiprocessing
import cv2
import os
import requests
import time
import signal
import sys
import keyboard
from datetime import datetime
from gtts import gTTS
from sic_framework.devices.desktop import Desktop
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, Dialogflow)
from sic_framework.core import utils_cv2
from sic_framework.core.message_python2 import BoundingBoxesMessage, CompressedImageMessage
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf
from sic_framework.services.face_detection.face_detection import FaceDetection

# Create folder for saving audio responses if it doesn't exist
os.makedirs("response_audio", exist_ok=True)

# Global variables
local_ai_response = None
language = "en"
imgs_buffer = multiprocessing.Queue(maxsize=1)
faces_buffer = multiprocessing.Queue(maxsize=1)
mic_initialized = False
camera_initialized = False

# Flags to stop processes
audio_process_running = multiprocessing.Event()
video_process_running = multiprocessing.Event()

# Signal handler to stop processes
def signal_handler(sig, frame):
    print("Signal received, stopping processes...")
    audio_process_running.clear()
    video_process_running.clear()
    cv2.destroyAllWindows()
    sys.exit(0)

# Attach signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Function to generate response from local AI model
def generate_response(prompt):
    url = "http://127.0.0.1:11434/api/chat"
    body = {
        "model": "llama3.2:latest",
        "messages": [
            {"role": "system", "content": "You are a robot assistant and keep responses concise (four sentences max)."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        response = requests.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return data['message']['content']
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

# Dialogflow callback function to handle recognition result
def on_dialog(message):
    global local_ai_response
    if message.response and message.response.recognition_result.is_final:
        print("Transcript:", message.response.recognition_result.transcript)
        local_ai_response = generate_response(message.response.recognition_result.transcript)

        if local_ai_response:
            # Save audio response with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = f"response_audio/textToSpeech_{timestamp}.mp3"
            speech = gTTS(text=local_ai_response, lang=language, slow=False, tld="com.au")
            speech.save(audio_path)

# Audio process function to handle Dialogflow requests and responses indefinitely
def audio_process_function():
    global mic_initialized, audio_process_running, local_ai_response
    audio_process_running.set()  # Set the running flag to True
    if not mic_initialized:
        desktop_audio = Desktop()
        keyfile_json = json.load(open("robot_key.json"))
        conf = DialogflowConf(keyfile_json=keyfile_json, sample_rate_hertz=44100, language="en")
        dialogflow = Dialogflow(ip='localhost', conf=conf)
        dialogflow.connect(desktop_audio.mic)
        dialogflow.register_callback(on_dialog)
        print("Audio system ready.")
        mic_initialized = True

    conversation_id = np.random.randint(10000)
    try:
        turn = 0
        while audio_process_running.is_set():
            print(f" ----- Conversation turn {turn}")
            contexts_dict = {"name": 1}
            reply = dialogflow.request(GetIntentRequest(conversation_id, contexts_dict))
            print("Detected intent:", reply.intent)
            if reply.fulfillment_message:
                print("Reply:", reply.fulfillment_message)

            # Reset local_ai_response before waiting for the next response
            local_ai_response = None

            # Wait for a response from the local AI
            while local_ai_response is None:
                time.sleep(0.1)  # Small sleep to avoid busy waiting
            
            # Print the local AI response
            print("Local AI:", local_ai_response)

            # Introduce a small delay for readability and control
            time.sleep(1)
            turn += 1
            
    except KeyboardInterrupt:
        print("Stopping audio system.")
    finally:
        dialogflow.stop()
        print("Audio system stopped.")


# Function to process image messages from video process
def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

# Function to process face bounding boxes from video process
def on_faces(message: BoundingBoxesMessage):
    faces_buffer.put(message.bboxes)

# Video process function to handle face detection and display
def video_process_function():
    global camera_initialized, video_process_running
    video_process_running.set() 
    if not camera_initialized:
        conf = DesktopCameraConf(fx=1.0, fy=1.0, flip=1) 
        desktop_video = Desktop(camera_conf=conf)
        face_rec = FaceDetection()
        face_rec.connect(desktop_video.camera)
        desktop_video.camera.register_callback(on_image)
        face_rec.register_callback(on_faces)
        print("Video system ready.")
        camera_initialized = True

    target_fps = 15
    frame_delay = 1 / target_fps
    last_frame_time = time.time()

    while video_process_running.is_set():
        try:
            img = imgs_buffer.get(timeout=1)
            if img is None or img.size == 0:
                print("Warning: Empty frame received.")
                continue

            img_rgb = img   

            try:
                faces = faces_buffer.get(timeout=1)
            except queue.Empty:
                print("No face data received; skipping face processing.")
                faces = []

            for face in faces:
                utils_cv2.draw_bbox_on_image(face, img_rgb)

            cv2.imshow('Face Detection', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Frame rate control
            current_time = time.time()
            if current_time - last_frame_time < frame_delay:
                time.sleep(frame_delay - (current_time - last_frame_time))
            last_frame_time = current_time
            
        except queue.Empty:
            print("No frames received; retrying...")
            time.sleep(0.1)
        except Exception as e:
            print(f"Exception in video process: {e}")
            time.sleep(0.5)

    cv2.destroyAllWindows()
    print("Video system stopped.")


# Main execution to start audio and video processes
if __name__ == "__main__":
    audio_process = multiprocessing.Process(target=audio_process_function, daemon=True)
    video_process = multiprocessing.Process(target=video_process_function, daemon=True)

    audio_process.start()
    video_process.start()

    print("Press 'q' to quit the program...")
    while True:
        if keyboard.is_pressed('q'):
            signal_handler(None, None)  # Trigger the signal handler to cleanly exit
            break

    audio_process.join()
    video_process.join()
