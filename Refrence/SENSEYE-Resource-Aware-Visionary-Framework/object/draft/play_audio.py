import subprocess
import os


def set_default_sink_if_available(desired_sink_name):
    """
    Set the default audio output device to the desired sink if it's available and not already selected.
    """
    try:
        # Get the current default sink
        result = subprocess.run(["pactl", "get-default-sink"], stdout=subprocess.PIPE, text=True, check=True)
        current_sink = result.stdout.strip()
        print(f"Current default sink: {current_sink}")

        # Check if the desired sink is already set
        if current_sink == desired_sink_name:
            print(f"{desired_sink_name} is already the default sink. Skipping.")
            return

        # List all available sinks
        result = subprocess.run(["pactl", "list", "short", "sinks"], stdout=subprocess.PIPE, text=True, check=True)
        available_sinks = [line.split("\t")[1] for line in result.stdout.splitlines()]
        print(f"Available sinks: {available_sinks}")

        # Check if the desired sink is available
        if desired_sink_name in available_sinks:
            # Set the desired sink as the default
            subprocess.run(["pactl", "set-default-sink", desired_sink_name], check=True)
            print(f"Default sink set to: {desired_sink_name}")
        else:
            print(f"{desired_sink_name} is not available. Skipping.")
    except subprocess.CalledProcessError as e:
        print(f"Error managing default sink: {e}")

def play_audio(audio_file):
    """
    Play an audio file using ffplay with the correct PulseAudio configuration.
    """
    try:
        # Ensure PulseAudio environment is set correctly
        os.environ["PULSE_SERVER"] = "/run/user/1000/pulse/native"
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

        # Play the audio file
        subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to play audio: {e}")


if __name__ == "__main__":
    # Set the default sink to Jabra
    sink_name = "alsa_output.usb-GN_Netcom_A_S_Jabra_EVOLVE_20_MS_A009E07823660A-00.analog-stereo"
    set_default_sink_if_available(sink_name)

    # Play the audio file
    play_audio("object_detected.mp3")
