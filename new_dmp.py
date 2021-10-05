from random import random
from threading import Thread
import time
import rospy
from std_msgs.msg import String

from functools import partial
from bokeh.plotting import curdoc, figure

from tornado import gen

import pickle
import numpy as np
from visualization import *
from processing import *

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Button, RangeSlider, Panel, Tabs, Legend, LegendItem, Span, RadioButtonGroup, Spinner

import json
from bokeh.embed import server_document 

class DMP:

    def __init__(self, doc):
        self.doc = curdoc()

        state_dict = pickle.load( open( '/home/sony/Documents/iam-web/iam-bokeh-server/franka_traj.pkl', "rb" ) )

        self.cartesian_trajectory = {}
        self.skill_state_dict = {}
        self.colors_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']

        for key in state_dict.keys():
            skill_dict = state_dict[key]
            
            if skill_dict["skill_description"] == "GuideMode":
                self.skill_state_dict = skill_dict["skill_state_dict"]
                self.cartesian_trajectory = process_cartesian_trajectories(self.skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
                
        self.trajectory_start_time = 0.0
        self.trajectory_end_time = self.skill_state_dict['time_since_skill_started'][-1]

        self.x_range = [self.trajectory_start_time, self.trajectory_end_time]

        self.truncated_start_time = self.trajectory_start_time
        self.truncated_end_time = self.trajectory_end_time

        self.truncation_threshold = Spinner(title="Truncation Threshold", low=0, high=1, step=0.005, value=0.005)

        (self.truncated_start_time, self.truncated_end_time) = get_start_and_end_times(self.skill_state_dict['time_since_skill_started'], self.cartesian_trajectory, self.skill_state_dict['q'], self.truncation_threshold.value)

        # add a button widget and configure with the call back
        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_callback)
        
        self.continue_button = Button(label="Continue")
        self.continue_button.on_click(self.continue_callback)

        self.done_button = Button(label="Done")
        self.done_button.on_click(self.done_callback)

        # add a button widget and configure with the call back
        self.truncate_button = Button(label="Automatically Truncate")
        self.truncate_button.on_click(self.truncate_callback)
        self.time_range = RangeSlider(title="Truncated Time", value=(self.truncated_start_time,self.truncated_end_time), start=self.trajectory_start_time, end=self.trajectory_end_time, step=0.01)

        self.skill_name = TextInput(title="Skill Name", value='')
        self.num_basis = Slider(title="Number of Basis Functions", value=4, start=1, end=10, step=1)

        self.dmp_type = RadioButtonGroup(labels=["cartesian", "joint"], active=0)

        self.start_time_span = Span(location=self.truncated_start_time,
                              dimension='height', line_color='green',
                              line_width=3)

        self.end_time_span = Span(location=self.truncated_end_time,
                              dimension='height', line_color='red',
                              line_width=3)

        self.time_range.js_link('value', self.start_time_span, 'location', attr_selector=0)
        self.time_range.js_link('value', self.end_time_span, 'location', attr_selector=1)

    def continue_callback(self):
        self.doc.clear()

        xs = np.tile(self.skill_state_dict['time_since_skill_started'].reshape((1,-1)), (7, 1)).tolist()

        p1 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys1 = np.transpose(self.cartesian_trajectory).tolist()
        r1 = p1.multi_line(xs=xs, ys=ys1, color=self.colors_list, line_width=3)

        p2 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys2 = np.transpose(self.cartesian_trajectory - self.cartesian_trajectory[0,:]).tolist()
        r2 = p2.multi_line(xs=xs, ys=ys2, color=self.colors_list, line_width=3)

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
        p2.add_layout(pose_legend)
        
        # Set up layouts and add to document
        inputs = column(self.dmp_type, self.skill_name, self.num_basis, self.submit_button)
        plots = column(p1,p2)
        self.doc.add_root(column(row(inputs, plots), self.done_button))

    def submit_callback(self):
        pass

    def done_callback(self):
        self.doc.clear()

    def truncate_callback(self):
        (self.truncated_start_time, self.truncated_end_time) = get_start_and_end_times(self.skill_state_dict['time_since_skill_started'], self.cartesian_trajectory, self.skill_state_dict['q'], self.truncation_threshold.value)
        self.start_time_span.location = self.truncated_start_time
        self.end_time_span.location = self.truncated_end_time
        
    def handle_dmp_training_request(self, request):
        xs = np.tile(self.skill_state_dict['time_since_skill_started'].reshape((1,-1)), (7, 1)).tolist()

        p1 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys1 = np.transpose(self.cartesian_trajectory).tolist()
        r1 = p1.multi_line(xs=xs, ys=ys1, color=self.colors_list, line_width=3)

        p2 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys2 = np.transpose(self.cartesian_trajectory - self.cartesian_trajectory[0,:]).tolist()
        r2 = p2.multi_line(xs=xs, ys=ys2, color=self.colors_list, line_width=3)

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
        p1.add_layout(self.start_time_span)
        p1.add_layout(self.end_time_span)
        p2.add_layout(pose_legend)
        p2.add_layout(self.start_time_span)
        p2.add_layout(self.end_time_span)
        tab1 = Panel(child=column(p1, row(self.truncation_threshold, self.truncate_button), self.time_range, self.continue_button, sizing_mode='scale_width'), title="Cartesian Pose")
        tab2 = Panel(child=column(p2, row(self.truncation_threshold, self.truncate_button), self.time_range, self.continue_button, sizing_mode='scale_width'), title="Relative Cartesian Pose")

        p3 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys3 = np.transpose(self.skill_state_dict['q']).tolist()
        r3 = p3.multi_line(xs=xs, ys=ys3, color=self.colors_list, line_width=3)

        p4 = figure(plot_width=1000, plot_height=300, x_range=self.x_range)
        ys4 = np.transpose(self.skill_state_dict['q']-self.skill_state_dict['q'][0,:]).tolist()
        r4 = p4.multi_line(xs=xs, ys=ys4, color=self.colors_list, line_width=3)

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
        p3.add_layout(self.start_time_span)
        p3.add_layout(self.end_time_span)
        p4.add_layout(joint_legend)
        p4.add_layout(self.start_time_span)
        p4.add_layout(self.end_time_span)
        tab3 = Panel(child=column(p3, row(self.truncation_threshold, self.truncate_button), self.time_range, self.continue_button, sizing_mode='scale_width'), title="Joint Position")
        tab4 = Panel(child=column(p4, row(self.truncation_threshold, self.truncate_button), self.time_range, self.continue_button, sizing_mode='scale_width'), title="Relative Joint Position")

        self.doc.add_root(Tabs(tabs=[tab1, tab2, tab3, tab4]))