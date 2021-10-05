
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
import helpers
import time

rospy.init_node('bokeh_server')

rospy.wait_for_service('dextr')
bridge = CvBridge()

dextr_client = rospy.ServiceProxy('dextr', DEXTRRequest)

pub = rospy.Publisher('masks', Image, queue_size=10)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

im = cv2.imread('/home/sony/new_yolo_food_photos/image3.png')

M, N, _ = im.shape
img = np.empty((M, N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((M, N, 4))
view[:,:,0] = im[:,:,2] # copy red channel
view[:,:,1] = im[:,:,1] # copy blue channel
view[:,:,2] = im[:,:,0] # copy green channel
view[:,:,3] = 255

img = img[::-1] # flip for Bokeh

TOOLS = "tap"
p = figure(title='Label the Image by tapping on 4 extreme points. You can press on a point again to remove it. Then type in the name of the object and press submit. Press Done when you have finished labeling.',
           tools=TOOLS,width=640,height=480,
           x_axis_type=None, y_axis_type=None)
img_source = ColumnDataSource({'image': [img]})

p.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=img_source)

source = ColumnDataSource(data=dict(x=[], y=[]))   
p.circle(source=source,x='x',y='y', radius=0.15, alpha=0.5, fill_color='red') 

object_names = []
masks = []
coordList=[]

#add a dot where the click happened
def callback(event):
    Coords=(event.x,event.y)

    removed_point = False

    for i in coordList:
        #print(np.sqrt(np.square(i[0] - Coords[0]) + np.square(i[1] - Coords[1])))
        if np.sqrt(np.square(i[0] - Coords[0]) + np.square(i[1] - Coords[1])) < 0.25:
            coordList.remove(i)
            removed_point = True
            break
    if not removed_point:
        coordList.append(Coords) 
    source.data = dict(x=[i[0] for i in coordList], y=[i[1] for i in coordList]) 

p.on_event(Tap, callback)

def redo_callback():
    masks.pop()
    if len(masks) == 0:
        img_source.data = {'image': [img]}
    else:
        result_image = helpers.overlay_masks(im / 255, masks)

        int_result_image = np.array(result_image*255).astype(np.int)

        cv2.imwrite('result.jpg', int_result_image)

        # Plot the results
        new_img = np.empty((M, N), dtype=np.uint32)
        new_view = new_img.view(dtype=np.uint8).reshape((M, N, 4))
        new_view[:,:,0] = int_result_image[:,:,2] # copy red channel
        new_view[:,:,1] = int_result_image[:,:,1] # copy blue channel
        new_view[:,:,2] = int_result_image[:,:,0] # copy green channel
        new_view[:,:,3] = 255

        new_img = new_img[::-1] # flip for Bokeh

        img_source.data = {'image': [new_img]}

    page_layout.children[2] = submit_button

# add a button widget and configure with the call back
redo_button = Button(label="Redo")
redo_button.on_click(redo_callback)

def continue_callback():
    coordList.clear()
    object_names.append(text.value)
    text.value = ''
    source.data = dict(x=[], y=[])

    page_layout.children[2] = submit_button

# add a button widget and configure with the call back
continue_button = Button(label="Continue")
continue_button.on_click(continue_callback)

def submit_callback():
    if len(coordList) == 4:    
        p.title.text = 'Label the Image by tapping on 4 extreme points. You can press on a point again to remove it. Then type in the name of the object and press submit. Press Done when you have finished labeling.'
        
        points = []
        for i in coordList:
            points.append(Point2D(int(i[0]*(N/10.0)), int(M-i[1]*(M/10.0))))

        sensor_image = bridge.cv2_to_imgmsg(im, "bgr8")

        resp = dextr_client(sensor_image, points)

        masks.append(np.array(bridge.imgmsg_to_cv2(resp.mask) > 0))

        result_image = helpers.overlay_masks(im / 255, masks)

        int_result_image = np.array(result_image*255).astype(np.int)

        #cv2.imwrite('result.jpg', int_result_image)

        # Plot the results
        new_img = np.empty((M, N), dtype=np.uint32)
        new_view = new_img.view(dtype=np.uint8).reshape((M, N, 4))
        new_view[:,:,0] = int_result_image[:,:,2] # copy red channel
        new_view[:,:,1] = int_result_image[:,:,1] # copy blue channel
        new_view[:,:,2] = int_result_image[:,:,0] # copy green channel
        new_view[:,:,3] = 255

        new_img = new_img[::-1] # flip for Bokeh

        img_source.data = {'image': [new_img]}

        page_layout.children[2] = row(redo_button, continue_button)

        #doc.clear()

    else:
        p.title.text = 'Please click on 4 extreme points before pressing the Submit Button.'

# add a button widget and configure with the call back
submit_button = Button(label="Submit")
submit_button.on_click(submit_callback)

def done_callback():
    doc.clear()
    # coord_string = ''
    # for i in actual_coord:
    #     coord_string += '('+str(i[0]) + ','+str(i[1])+'),'
    # pub.publish(coord_string)

# add a button widget and configure with the call back
done_button = Button(label="Done")
done_button.on_click(done_callback)

text = TextInput(title="Object Name", value='')

#layout=Column(p)
page_layout=column(p,text, submit_button, done_button)

curdoc().add_root(page_layout)

@gen.coroutine
def ros_update(data):
    plot.title.text = data
    
def ros_handler():

    i = 1
    while not rospy.is_shutdown():
        try:
            string_data = rospy.wait_for_message('/chatter', String, timeout=1)
            doc.add_next_tick_callback(partial(ros_update, data=string_data.data))
        except:
            pass
        rospy.sleep(0.01)
        i+=1

thread = Thread(target=ros_handler)
thread.start()
