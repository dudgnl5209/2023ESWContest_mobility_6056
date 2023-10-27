from ublox_gps import UbloxGps
from flask import Flask, render_template, Response
from time import sleep
from gtts import gTTS
import pygame
import cv2
import numpy as np
import serial

app = Flask(__name__)

serial_port = serial.Serial('/dev/ttyACM0', baudrate=38400, timeout=1)
gps_device = UbloxGps(serial_port)

front = cv2.VideoCapture(0)
front.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
front.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

right = cv2.VideoCapture(4)
right.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

tts = gTTS('신호 위반 발생', lang = 'ko')
tts.save('notify.mp3')

pygame.mixer.init()
pygame.mixer.music.load('notify.mp3')

def GenerateFrontFrames():
    while True:
        ref, frame = front.read()
        if not ref:
            break
        else:
            ref, buffer = cv2.imencode('.jpg', frame)
            output = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')

def GenerateRightFrames():
    while True:
        ref, frame = right.read()
        if not ref:
            break
        else:
            ref, buffer = cv2.imencode('.jpg', frame)
            output = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')


@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/front_stream')
def FrontStream():
    return Response(GenerateFrontFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/right_stream')
def RightStream():
    return Response(GenerateRightFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gps')
def Gps():
    try:
        coords = gps_device.geo_coords()
        print(coords.lon, coords.lat)
    except (ValueError, IOError) as err:
        print(err)
    return render_template('gps.html', lat=coords.lat, lon=coords.lon, head=coords.headMot)

@app.route('/notify')
def Notify():
    pygame.mixer.music.play()
    sleep(1)
    return render_template('notify.html')

if __name__ == "__main__":
    app.run(host="localhost", port="8080")
    serial_port.close()
