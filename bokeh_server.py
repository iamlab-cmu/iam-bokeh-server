#!/usr/bin/env python
from threading import Thread
import time
import rospy

from functools import partial
from bokeh.plotting import curdoc, figure

from tornado import gen

import pickle
import numpy as np
from visualization import *
from processing import *

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Button, RangeSlider, Panel, Tabs, Legend, LegendItem, Span, RadioButtonGroup, Spinner

from bokeh_server_msgs.msg import *

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column, Button, TextInput
from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.layouts import column, row

import cv2

from sensor_msgs.msg import Image

from dextr_msgs.srv import DEXTRRequest,DEXTRRequestResponse
from dextr_msgs.msg import Point2D
import helpers

from new_dextr import DEXTR
from new_dmp import DMP

rospy.init_node('bokeh_server')

pub = rospy.Publisher('bokeh_response', Response, queue_size=10)
rospy.wait_for_service('dextr')

dextr_client = rospy.ServiceProxy('dextr', DEXTRRequest)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

dextr = DEXTR(doc)
dmp = DMP(doc)

@gen.coroutine
def handle_request(msg):
    doc.clear()
    if msg.display_type == 0:
        dmp.handle_dmp_training_request(msg)
    elif msg.display_type == 1:
        dextr.handle_dextr_request(msg)

def ros_handler():

    while not rospy.is_shutdown():
        try:
            request_msg = rospy.wait_for_message('/bokeh_request', Request, timeout=1)
            doc.add_next_tick_callback(partial(handle_request, msg=request_msg))
        except:
            pass
        rospy.sleep(1)

thread = Thread(target=ros_handler)
thread.start()