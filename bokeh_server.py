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
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

import helpers

from new_dextr import DEXTR
from new_dmp import DMP
from point_goals import PointGoal

rospy.init_node('bokeh_server')

pub = rospy.Publisher('bokeh_response', Response, queue_size=10)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

bridge = CvBridge()

dmp = DMP(doc, pub)
dextr = DEXTR(doc, pub, bridge)
point_goal = PointGoal(doc, pub, bridge)

@gen.coroutine
def handle_request(msg):
    doc.clear()
    print("Received message with display_type" + str(msg.display_type))
    if msg.display_type == 0:
        dmp.handle_dmp_training_request(msg)
    elif msg.display_type == 1:
        dextr.handle_dextr_request(msg)
    elif msg.display_type == 2:
        point_goal.handle_point_goal_request(msg)

def ros_handler():

    while not rospy.is_shutdown():
        try:
            request_msg = rospy.wait_for_message('/bokeh_request', Request, timeout=1)
            doc.add_next_tick_callback(partial(handle_request, msg=request_msg))
        except:
            pass

thread = Thread(target=ros_handler)
thread.start()