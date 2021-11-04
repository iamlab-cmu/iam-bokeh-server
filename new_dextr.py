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

from dextr_msgs.srv import DEXTRRequest,DEXTRRequestResponse
from dextr_msgs.msg import Point2D
from bokeh_server_msgs.msg import Response
import helpers
import time

class DEXTR:

    def __init__(self, doc, pub):
        self.doc = doc
        self.pub = pub
        self.bridge = CvBridge()

        print('Waiting for DEXTR Service.')
        rospy.wait_for_service('dextr')

        print('DEXTR Service is Ready.')
        self.dextr_client = rospy.ServiceProxy('dextr', DEXTRRequest)

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
        self.redo_button = Button(label="Redo")
        self.redo_button.on_click(self.redo_callback)

        # add a button widget and configure with the call back
        self.continue_button = Button(label="Continue")
        self.continue_button.on_click(self.continue_callback)

        # add a button widget and configure with the call back
        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_callback)

        # add a button widget and configure with the call back
        self.done_button = Button(label="Done")
        self.done_button.on_click(self.done_callback)

        self.text = TextInput(title="Object Name", value='')

        self.object_names = []
        self.masks = []
        self.coordList=[]

        print('Done Initializing DEXTR')

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

    def redo_callback(self):
        self.masks.pop()
        if len(self.masks) == 0:
            self.img_source.data = {'image': [self.img]}
        else:
            result_image = helpers.overlay_masks(self.im / 255, self.masks)

            int_result_image = np.array(result_image*255).astype(np.int)

            # Plot the results
            new_img = np.empty((self.M, self.N), dtype=np.uint32)
            new_view = new_img.view(dtype=np.uint8).reshape((self.M, self.N, 4))
            new_view[:,:,0] = int_result_image[:,:,2] # copy red channel
            new_view[:,:,1] = int_result_image[:,:,1] # copy blue channel
            new_view[:,:,2] = int_result_image[:,:,0] # copy green channel
            new_view[:,:,3] = 255

            new_img = new_img[::-1] # flip for Bokeh

            self.img_source.data = {'image': [new_img]}

        self.page_layout.children[2] = self.submit_button

    def continue_callback(self):
        self.coordList.clear()
        self.object_names.append(self.text.value)
        self.text.value = ''
        self.source.data = dict(x=[], y=[])

        self.page_layout.children[2] = self.submit_button

    def done_callback(self):
        self.doc.clear()

    def submit_callback(self):
        if len(self.coordList) == 4:    
            self.p.title.text = ''
            
            points = []
            for i in self.coordList:
                points.append(Point2D(int(i[0]*(self.N/10.0)), int(self.M-i[1]*(self.M/10.0))))

            sensor_image = self.bridge.cv2_to_imgmsg(self.im, "bgr8")

            resp = self.dextr_client(sensor_image, points)

            self.masks.append(np.array(self.bridge.imgmsg_to_cv2(resp.mask) > 0))

            result_image = helpers.overlay_masks(self.im / 255, self.masks)

            int_result_image = np.array(result_image*255).astype(np.int)

            # Plot the results
            new_img = np.empty((self.M, self.N), dtype=np.uint32)
            new_view = new_img.view(dtype=np.uint8).reshape((self.M, self.N, 4))
            new_view[:,:,0] = int_result_image[:,:,2] # copy red channel
            new_view[:,:,1] = int_result_image[:,:,1] # copy blue channel
            new_view[:,:,2] = int_result_image[:,:,0] # copy green channel
            new_view[:,:,3] = 255

            new_img = new_img[::-1] # flip for Bokeh

            self.img_source.data = {'image': [new_img]}

            self.page_layout.children[2] = row(self.redo_button, self.continue_button)
        else:
            self.p.title.text = 'Please click on 4 extreme points before pressing the Submit Button.'

    def handle_dextr_request(self, request):

        self.object_names = []
        self.masks = []
        self.coordList=[]

        print('Received Image')
        try:
            self.im = self.bridge.imgmsg_to_cv2(request.image)

            print('Converted Image')
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

            self.page_layout=column(self.p, self.text, self.submit_button, self.done_button)

            self.doc.add_root(self.page_layout)
        except Exception as e: 
            print(e)
