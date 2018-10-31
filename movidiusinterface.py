#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

# Python script to open and close a single NCS device

import mvnc.mvncapi as fx


# main entry point for the program
class MovidiusInterface():
    def __init__(self):
        # set the logging level for the NC API
        fx.SetGlobalOption(fx.GlobalOption.LOG_LEVEL, 0)
        # get a list of names for all the devices plugged into the system
        ncs_names = fx.EnumerateDevices()
        if (len(ncs_names) < 1):
            raise Exception("Error - no NCS devices detected, verify an NCS device is connected.")

        # get the first NCS device by its name.  For this program we will always open the first NCS device.
        self.dev = fx.Device(ncs_names[0])

        # try to open the device.  this will throw an exception if someone else has it open already
        try:
            self.dev.OpenDevice()
            print("Hello NCS! Device opened normally.")
        except:
            raise Exception("Error - Could not open NCS device.")

    def __del__(self):
        try:
            self.dev.CloseDevice()
            print("Goodbye NCS! Device closed normally.")
            print("NCS device working.")
        except:
            raise Exception("Error - could not close NCS device.")
