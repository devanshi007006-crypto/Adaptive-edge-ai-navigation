#!/usr/bin/python

import Jetson.GPIO as GPIO
import serial
import time

ser = serial.Serial('/dev/ttyTHS1',115200)
ser.flushInput()

powerKey = 6
rec_buff = ''
rec_buff2 = ''
time_count = 0

def sendAt(command,back,timeout):
	#rec_buff = ''
	ser.write((command+'\r\n').encode())
	time.sleep(timeout)
	if ser.inWaiting():
		time.sleep(0.01 )
		rec_buff = ser.read(ser.inWaiting())
	if rec_buff != '':
		if back not in rec_buff.decode():
			print(command + ' ERROR')
			print(command + ' back:\t' + rec_buff.decode())
			return 0
		else:
			print(rec_buff.decode())
			return 1
	else:
		print('GPS is not ready')
		return 0

def getGpsPosition():
	rec_null = True
	answer = 0
	print('Start GPS session...')
	rec_buff = ''
	sendAt('AT+CGPS=1,2','OK',1)
	time.sleep(2)
	while rec_null:
		answer = sendAt('AT+CGPSINFO','+CGPSINFO: ',1)
		if 1 == answer:
                    answer = 0
                    if ',,,,,,,,' in rec_buff:
                        print('GPS is not ready,wait 10 seconds')
                        rec_null = False
                        time.sleep(10)
		else:
                    print('error %d'%answer)
                    rec_buff = ''
                    sendAt('AT+CGPS=0','OK',1)
                    return False
		time.sleep(1.5)

def powerOn(powerKey):
	print('SIM7600X is starting:')
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(powerKey,GPIO.OUT)
	time.sleep(0.1)
	GPIO.output(powerKey,GPIO.HIGH)
	time.sleep(2)
	GPIO.output(powerKey,GPIO.LOW)
	time.sleep(20)
	ser.flushInput()
	print('SIM7600X is ready')

def powerDown(powerKey):
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(powerKey,GPIO.OUT)
	print('SIM7600X is loging off:')
	GPIO.output(powerKey,GPIO.HIGH)
	time.sleep(3)
	GPIO.output(powerKey,GPIO.LOW)
	time.sleep(18)
	print('Good bye')

def checkStart():
        while True:
            ser.write( ('AT\r\n').encode() )
            time.sleep(0.1);
            if ser.inWaiting():
                time.sleep(0.01)
                recBuff = ser.read(ser.inWaiting())
                print( 'try to start\r\n' + recBuff.decode() )
                if 'OK' in recBuff.decode():
                    recBuff = ''
                    return
            else:
                powerOn(powerKey)
                time.sleep(1)
try:
        checkStart()
        getGpsPosition()
        powerDown(powerKey)
except:
	if ser != None:
		ser.close()
	powerDown(powerKey)
	GPIO.cleanup()
if ser != None:
        ser.close()
        GPIO.cleanup()	
