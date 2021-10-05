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

rospy.init_node('bokeh_server')

pub = rospy.Publisher('bokeh_response', Response, queue_size=10)
rospy.wait_for_service('dextr')

dextr_client = rospy.ServiceProxy('dextr', DEXTRRequest)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

dextr = DEXTR(doc)

# # Set up callbacks
# def update_title(attrname, old, new):
#     plot.title.text = text.value

# text.on_change('value', update_title)

# for w in [start_time, end_time, phase, freq]:
#     w.on_change('value', update_data)

state_dict = pickle.load( open( '/home/sony/Documents/iam-web/iam-bokeh-server/franka_traj.pkl', "rb" ) )

cartesian_trajectory = {}
skill_state_dict = {}

for key in state_dict.keys():
    skill_dict = state_dict[key]
    
    if skill_dict["skill_description"] == "GuideMode":
        skill_state_dict = skill_dict["skill_state_dict"]
        cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
        
trajectory_start_time = 0.0
trajectory_end_time = skill_state_dict['time_since_skill_started'][-1]


def handle_dmp_training_request(msg):

    x_range = [trajectory_start_time, trajectory_end_time]

    truncated_start_time = trajectory_start_time
    truncated_end_time = trajectory_end_time

    def submit_callback():
        doc.clear()
        # Set up data
        N = 200
        x = np.linspace(0, 4*np.pi, N)
        y = np.sin(x)
        source = ColumnDataSource(data=dict(x=x, y=y))


        # Set up plot
        plot = figure(height=400, width=400, title="my sine wave",
                  tools="crosshair,pan,reset,save,wheel_zoom",
                  x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

        plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

        # Set up callbacks
        def update_title(attrname, old, new):
            plot.title.text = text.value

        text = TextInput(title="Skill Name", value='my skill name')
        start_time = Slider(title="start_time", value=0.0, start=-5.0, end=5.0, step=0.1)
        end_time = Slider(title="end_time", value=1.0, start=-5.0, end=5.0, step=0.1)
        phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
        freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)
        button = Button(label="Submit")
        button.on_click(submit_callback)

        text.on_change('value', update_title)

        radio_button_group = RadioButtonGroup(labels=["cartesian", "joint"], active=0)
        radio_button_group.js_on_click(CustomJS(code="""
            console.log('radio_button_group: active=' + this.active, this.toString())
        """))
        
        # Set up layouts and add to document
        inputs = column(radio_button_group, text, start_time, end_time, phase, freq, button)
        doc.add_root(column(inputs, plot))

    # add a button widget and configure with the call back
    submit_button = Button(label="Submit")
    submit_button.on_click(submit_callback)

    def truncate_callback():
        (truncated_start_time, truncated_end_time) = get_start_and_end_times(skill_state_dict['time_since_skill_started'], cartesian_trajectory, skill_state_dict['q'], truncation_threshold.value)
        start_time_span.location = truncated_start_time
        end_time_span.location = truncated_end_time

    truncation_threshold = Spinner(title="Truncation Threshold", low=0, high=1, step=0.005, value=0.005)

    (truncated_start_time, truncated_end_time) = get_start_and_end_times(skill_state_dict['time_since_skill_started'], cartesian_trajectory, skill_state_dict['q'], truncation_threshold.value)

    # add a button widget and configure with the call back
    truncate_button = Button(label="Automatically Truncate")
    truncate_button.on_click(truncate_callback)
    time_range = RangeSlider(title="Truncated Time", value=(truncated_start_time,truncated_end_time), start=trajectory_start_time, end=trajectory_end_time, step=0.01)

    start_time_span = Span(location=truncated_start_time,
                          dimension='height', line_color='green',
                          line_width=3)

    end_time_span = Span(location=truncated_end_time,
                          dimension='height', line_color='red',
                          line_width=3)
    time_range.js_link('value', start_time_span, 'location', attr_selector=0)
    time_range.js_link('value', end_time_span, 'location', attr_selector=1)

    xs = np.tile(skill_state_dict['time_since_skill_started'].reshape((1,-1)), (7, 1)).tolist()
    colors_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']

    p1 = figure(plot_width=1000, plot_height=300, x_range=x_range)
    ys1 = np.transpose(cartesian_trajectory).tolist()
    r1 = p1.multi_line(xs=xs, ys=ys1, color=colors_list, line_width=3)

    p2 = figure(plot_width=1000, plot_height=300, x_range=x_range)
    ys2 = np.transpose(cartesian_trajectory - cartesian_trajectory[0,:]).tolist()
    r2 = p2.multi_line(xs=xs, ys=ys2, color=colors_list, line_width=3)

    pose_legend = Legend(items=[
        LegendItem(label='x', renderers=[r1,r2], index=0),
        LegendItem(label="y", renderers=[r1,r2], index=1),
        LegendItem(label="z", renderers=[r1,r2], index=2),
        LegendItem(label="qw", renderers=[r1,r2], index=3),
        LegendItem(label="qx", renderers=[r1,r2], index=4),
        LegendItem(label="qy", renderers=[r1,r2], index=5),
        LegendItem(label="qz", renderers=[r1,r2], index=6),
    ], click_policy="mute")
    p1.add_layout(pose_legend)
    p1.add_layout(start_time_span)
    p1.add_layout(end_time_span)
    p2.add_layout(pose_legend)
    p2.add_layout(start_time_span)
    p2.add_layout(end_time_span)
    tab1 = Panel(child=column(p1, row(truncation_threshold, truncate_button), time_range, submit_button, sizing_mode='scale_width'), title="Cartesian Pose")
    tab2 = Panel(child=column(p2, row(truncation_threshold, truncate_button), time_range, submit_button, sizing_mode='scale_width'), title="Relative Cartesian Pose")

    p3 = figure(plot_width=1000, plot_height=300, x_range=x_range)
    ys3 = np.transpose(skill_state_dict['q']).tolist()
    r3 = p3.multi_line(xs=xs, ys=ys3, color=colors_list, line_width=3)

    p4 = figure(plot_width=1000, plot_height=300, x_range=x_range)
    ys4 = np.transpose(skill_state_dict['q']-skill_state_dict['q'][0,:]).tolist()
    r4 = p4.multi_line(xs=xs, ys=ys4, color=colors_list, line_width=3)

    joint_legend = Legend(items=[
        LegendItem(label='1', renderers=[r3,r4], index=0),
        LegendItem(label="2", renderers=[r3,r4], index=1),
        LegendItem(label="3", renderers=[r3,r4], index=2),
        LegendItem(label="4", renderers=[r3,r4], index=3),
        LegendItem(label="5", renderers=[r3,r4], index=4),
        LegendItem(label="6", renderers=[r3,r4], index=5),
        LegendItem(label="7", renderers=[r3,r4], index=6),
    ], click_policy="mute")
    p3.add_layout(joint_legend)
    p3.add_layout(start_time_span)
    p3.add_layout(end_time_span)
    p4.add_layout(joint_legend)
    p4.add_layout(start_time_span)
    p4.add_layout(end_time_span)
    tab3 = Panel(child=column(p3, row(truncation_threshold, truncate_button), time_range, submit_button, sizing_mode='scale_width'), title="Joint Position")
    tab4 = Panel(child=column(p4, row(truncation_threshold, truncate_button), time_range, submit_button, sizing_mode='scale_width'), title="Relative Joint Position")

    doc.add_root(Tabs(tabs=[tab1, tab2, tab3, tab4]))

def handle_image_labeling_request(msg):
    dextr.handle_dextr_request(msg)


@gen.coroutine
def handle_request(msg):
    doc.clear()
    if msg.display_type == 0:
        handle_dmp_training_request(msg)
    elif msg.display_type == 1:
        handle_image_labeling_request(msg)

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