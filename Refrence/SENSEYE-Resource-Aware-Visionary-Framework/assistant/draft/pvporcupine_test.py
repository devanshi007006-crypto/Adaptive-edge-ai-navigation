import pvporcupine
from pvrecorder import PvRecorder

def main():
    # Your Picovoice access key
    access_key = "av8pDZ7dGqZViKkn0FRVrwhH/bqO8dVqpssKc0sJM5zDjU1RcCEpCg"  # Replace with your access key

    # Path to the hotword model (.ppn file)
    keyword_paths = ["hey-tapio.ppn"]  # Replace with the path to your .ppn file

    # Initialize Porcupine
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=keyword_paths
    )

    # Initialize PvRecorder
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()
    print("Listening for the hotword...")

    try:
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Hotword detected!")
                # Add your logic here (e.g., trigger a function)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        recorder.stop()
        recorder.delete()
        porcupine.delete()

if __name__ == "__main__":
    main()
