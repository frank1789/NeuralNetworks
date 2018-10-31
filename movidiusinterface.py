#! /usr/bin/env python3
#! -*- coding:utf-8 -*-

# Python script to open and close a single NCS device API v2
from mvnc import mvncapi


# main entry point for the program
class MovidiusInterface():
    def __init__(self):
        # set the logging level for the NC API
        mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, 0)
        # get a list of names for all the devices plugged into the system
        device_list = mvncapi.enumerate_devices()
        if not device_list:
            raise Exception("Error - No neural compute devices detected.")

        else:
            print(len(device_list), "neural compute devices found!")


        # Get a list of valid device identifiers
        device_list = mvncapi.enumerate_devices()
        # Create a Device instance for the first device found
        self.device = mvncapi.Device(device_list[0])
        # Open communication with the device
        # try to open the device.  this will throw an exception if someone else has it open already
        try:
            self.device.open()
            print("Hello NCS! Device opened normally.")
        except:
            raise Exception("Error - Could not open NCS device.")



    def __del__(self):
        try:
            # Close the device and destroy the device handle
            self.device.close()
            self.device.destroy()
            print("Goodbye NCS! Device closed normally.")
            print("NCS device working.")
        except:
            raise Exception("Error - could not close NCS device.")


a = MovidiusInterface()
