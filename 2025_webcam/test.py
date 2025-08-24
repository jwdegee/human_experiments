from psychopy.hardware import camera

mic = camera.Microphone(device=11)
cam = camera.Camera(device=16, mic=mic)

cam.open()
cam.record()  # starts recording
while cam.recordingTime < 10.0:  # record for 5 seconds
    cam.update()         
cam.stop()  # stops recording
cam.save('myVideo_trial.mp4', mergeAudio=False)
cam.close()