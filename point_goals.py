#!/usr/bin/env python

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column, Button, TextInput
from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.layouts import column, row

from threading import Thread
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from tornado import gen

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from functools import partial

from bokeh_server_msgs.msg import Response
from iam_common_msgs.msg import Int32Point2D

import helpers
import time

class PointGoal:

    def __init__(self, doc, pub, bridge):
        self.doc = doc
        self.pub = pub
        self.bridge = bridge

        self.M = 640
        self.N = 480
        self.img = np.empty((self.M, self.N), dtype=np.uint32)

        self.img = self.img[::-1] # flip for Bokeh

        self.img_source = ColumnDataSource({'image': [self.img]})

        self.source = ColumnDataSource(data=dict(x=[], y=[]))   

        self.TOOLS = "tap"

        self.p = figure(title='',
                   tools=self.TOOLS,width=640,height=480,
                   x_axis_type=None, y_axis_type=None)

        # add a button widget and configure with the call back
        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_callback)

        self.text = TextInput(title="Object Name", value='')

        self.object_names = []
        self.masks = []
        self.coordList=[]

    #add a dot where the click happened
    def tap_callback(self, event):
        Coords=(event.x,event.y)

        removed_point = False

        for i in self.coordList:
            if np.sqrt(np.square(i[0] - Coords[0]) + np.square(i[1] - Coords[1])) < 0.25:
                self.coordList.remove(i)
                removed_point = True
                break
        if not removed_point:
            self.coordList.append(Coords) 
        self.source.data = dict(x=[i[0] for i in self.coordList], y=[i[1] for i in self.coordList])


    def submit_callback(self):
        self.doc.clear()

        response_msg = Response()
        for i in self.coordList:
            response_msg.desired_positions.append(Int32Point2D(int(i[0]*(self.N/10.0)), int(self.M-i[1]*(self.M/10.0))))

        self.pub.publish(response_msg)

    def handle_point_goal_request(self, request):

        self.object_names = []
        self.coordList=[]

        self.im = self.bridge.imgmsg_to_cv2(request.image)
        self.M, self.N, _ = self.im.shape
        self.img = np.empty((self.M, self.N), dtype=np.uint32)

        view = self.img.view(dtype=np.uint8).reshape((self.M, self.N, 4))
        view[:,:,0] = self.im[:,:,2] # copy red channel
        view[:,:,1] = self.im[:,:,1] # copy blue channel
        view[:,:,2] = self.im[:,:,0] # copy green channel
        view[:,:,3] = 255

        self.img = self.img[::-1] # flip for Bokeh

        self.img_source = ColumnDataSource({'image': [self.img]})

        self.p.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=self.img_source)

        self.p.circle(source=self.source,x='x',y='y', radius=0.15, alpha=0.5, fill_color='red') 

        self.p.on_event(Tap, self.tap_callback)

        self.page_layout=column(self.p, self.text, self.submit_button)

        self.doc.add_root(self.page_layout)
