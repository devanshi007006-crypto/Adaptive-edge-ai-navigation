#!/bin/bash
#env
#source inference/bin/activate
  

# go to location
#cd jetson-inference/python/www/flask

# Loop to rerun the command if it fails
while true; do

	python3 jetson-inference/python/www/flask/app.py \
	--ssl-key=key.pem --ssl-cert=cert.pem \
    --classification=resnet18 \
    --detection=ssd-mobilenet-v2 \
    --segmentation=fcn-resnet18-mhp \
    --pose=resnet18-hand \
    --action=resnet18-kinetics \
    --background=u2net \
    --input=/dev/video6

	if [ $? -eq 0 ]; then
		break
	else
		echo "Command failed, retrying..."
		sleep 2  # Optional: wait for 2 seconds before retrying
	fi
done
