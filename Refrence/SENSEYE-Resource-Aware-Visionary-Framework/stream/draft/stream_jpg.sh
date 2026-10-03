#!/bin/bash
  
# go to location
#cd ~/jetson-inference/build/aarch64/bin

# Loop to rerun the command if it fails
while true; do
	video-viewer --headless --ssl-key=key.pem --ssl-cert=cert.pem /dev/video4 file://my_image.jpg
	if [ $? -eq 0 ]; then
		break
	else
		echo "Command failed, retrying..."
		sleep 2  # Optional: wait for 2 seconds before retrying
	fi
done

  
	#rtsp://@:8554/stream 
	#./video-viewer --ssl-key=key.pem --ssl-cert=cert.pem /dev/video6 	webrtc://@:8554/output
	#video-viewer /dev/video6 
	#	webrtc://@:8554/output
	# rtsp://@:8554/stream
