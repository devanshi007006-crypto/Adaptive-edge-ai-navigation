import Jetson.GPIO as GPIO
import time

PIN_BUTTON = 40

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_BUTTON, GPIO.IN)

try:
    while True:
        state = GPIO.input(PIN_BUTTON)
        print("Button state:", "Pressed" if state == GPIO.LOW else "Released")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    GPIO.cleanup()
