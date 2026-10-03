#!/bin/bash
#
#	--ssl-key=key.pem --ssl-cert=cert.pem 
# 	--host=192.168.193.100 --port=8050
#	--detection=ssd-mobilenet-v2 --pose=resnet18-hand --action=resnet18-kinetics
#	--resources=data/config.json



# go to location
cd ~/jetson-inference/python/www/dash

python3 app.py 


