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

from bokeh_server_msgs.msg import Response
from bokeh.layouts import column, row
from bokeh.core.properties import value
from bokeh.models import ColumnDataSource, Slider, TextInput, Button, RangeSlider, Panel, Tabs, Legend, LegendItem, Span, RadioButtonGroup, Spinner

from dmp_class import DMPTrajectory

class DMP:

    def __init__(self, doc, pub):
        self.doc = curdoc()
        self.pub = pub

        state_dict = pickle.load( open( '/home/sony/Documents/iam-web/iam-bokeh-server/franka_traj.pkl', "rb" ) )

        self.cartesian_trajectory = {}
        self.skill_state_dict = {}
        self.colors_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
        self.cart_legend_labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        self.joint_legend_labels = ['1', '2', '3', '4', '5', '6', '7']

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
        self.alpha = Slider(title="Alpha", value=5.0, start=0.0, end=20.0, step=0.1)

        self.top_plot_source = ColumnDataSource(data=dict(xs=[], ys=[]))  
        self.bot_plot_source = ColumnDataSource(data=dict(xs=[], ys=[]))  

        self.beta = self.alpha.value/4.0
        self.tau = 0.1
        self.num_sensors = 2
        self.num_dims = 7
        self.rollout_dt = 0.01

        self.dmp_type = RadioButtonGroup(labels=["cartesian", "joint"], active=0)
        self.dmp_type.on_change('active', lambda attr, old, new: self.radio_callback())

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

        (self.truncated_trajectory_times, self.truncated_cartesian_trajectory, self.truncated_joint_trajectory) = truncate_trajectory(self.skill_state_dict['time_since_skill_started'], self.cartesian_trajectory, self.skill_state_dict['q'], self.start_time_span.location, self.end_time_span.location)

        x_range = [0.0, self.truncated_trajectory_times[-1]]

        x = self.truncated_trajectory_times.tolist()

        self.p1 = figure(plot_width=1000, plot_height=300, x_range=x_range)
        ys1 = np.transpose(self.truncated_cartesian_trajectory - self.truncated_cartesian_trajectory[0,:]).tolist()

        self.top_plot_source.data = dict(x=x, y0=ys1[0], y1=ys1[1], y2=ys1[2], y3=ys1[3], y4=ys1[4], y5=ys1[5], y6=ys1[6])
        y_list = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6']
        for i in range(7):
            self.p1.line(x='x', y=y_list[i], source=self.top_plot_source, color=self.colors_list[i], alpha=0.8, line_width=3, legend_label=self.cart_legend_labels[i], muted_color=self.colors_list[i], muted_alpha=0.2)

        self.p1.legend.click_policy="mute"

        self.beta = self.alpha.value/4.0
        self.tau = 0.5/self.truncated_trajectory_times[-1]
        self.num_sensors = 2

        trajectory_times = self.truncated_trajectory_times.reshape(-1, 1)
        
        trajectory = self.truncated_cartesian_trajectory - self.truncated_cartesian_trajectory[0,:]

        traj_time = len(np.arange(0, self.truncated_trajectory_times[-1], self.rollout_dt))

        dmp_pos_traj = DMPTrajectory(self.tau, self.alpha.value, self.beta, 3, self.num_basis.value, self.num_sensors)
        weights_pos, _ = dmp_pos_traj.train_using_individual_trajectory('position', trajectory_times, trajectory[:, :3])
        self.quat_canonical_goal_time=(self.truncated_trajectory_times[-1]-1.0)
        self.quat_alpha_phase = self.truncated_trajectory_times[-1]
        quat_alpha, quat_num_basis = self.alpha.value, self.num_basis.value
        quat_beta = quat_alpha / 4.0
        dmp_quat_traj = DMPTrajectory(self.tau, quat_alpha, quat_beta, 3, quat_num_basis, 1, add_min_jerk=False, 
                                      alpha_phase=self.quat_alpha_phase)
        # Quaternion DMPs right now use dt to get canonical variable.
        weights_quat, _ = dmp_quat_traj.train_using_individual_trajectory(
            'quaternion', trajectory_times, trajectory[:, 3:], dt=self.rollout_dt)
        y_pos, dy_pos, x_pos, _, _ = dmp_pos_traj.run_dmp_with_weights(weights_pos,
                                                                   np.zeros((3)),
                                                                   self.rollout_dt,
                                                                   traj_time=traj_time)

        q, qd, qdd, q_xlog = dmp_quat_traj.run_quaternion_dmp_with_weights(
            weights_quat, 
            trajectory[0, 3:], 
            trajectory[-1, 3:], 
            self.rollout_dt,
            traj_time=traj_time,
            quat_canonical_goal_time=self.quat_canonical_goal_time
        )   

        self.p2 = figure(plot_width=1000, plot_height=300, x_range=x_range)

        new_trajectory_times = np.arange(y_pos.shape[0]) * self.rollout_dt
        self.bot_plot_source.data = dict(x=new_trajectory_times.tolist(), 
                                         y0=np.array(y_pos[:,0]).tolist(), y1=np.array(y_pos[:,1]).tolist(), y2=np.array(y_pos[:,2]).tolist(), 
                                         y3=np.array(q[:,0]).tolist(), y4=np.array(q[:,1]).tolist(), y5=np.array(q[:,2]).tolist(), y6=np.array(q[:,3]).tolist())
        for i in range(7):
            self.p2.line(x='x', y=y_list[i], source=self.bot_plot_source, color=self.colors_list[i], alpha=0.8, line_width=3, legend_label=self.cart_legend_labels[i], muted_color=self.colors_list[i], muted_alpha=0.2)

        self.p2.legend.click_policy="mute"
        
        # Set up layouts and add to document
        inputs = column(self.dmp_type, self.skill_name, self.num_basis, self.alpha, self.submit_button)
        plots = column(self.p1, self.p2)
        self.doc.add_root(column(row(inputs, plots), self.done_button))

    def radio_callback(self):
        print(self.dmp_type.active)
        if self.dmp_type.active == 0:        

            x = self.truncated_trajectory_times.tolist()
            ys1 = np.transpose(self.truncated_cartesian_trajectory - self.truncated_cartesian_trajectory[0,:]).tolist()
            self.top_plot_source.data = dict(x=x, y0=ys1[0], y1=ys1[1], y2=ys1[2], y3=ys1[3], y4=ys1[4], y5=ys1[5], y6=ys1[6])
            for i in range(7):
                self.p1.legend.items[i].update(label=value(self.cart_legend_labels[i]))

            trajectory_times = self.truncated_trajectory_times.reshape(-1, 1)
        
            trajectory = self.truncated_cartesian_trajectory - self.truncated_cartesian_trajectory[0,:]

            traj_time = len(np.arange(0, self.truncated_trajectory_times[-1], self.rollout_dt))

            dmp_pos_traj = DMPTrajectory(self.tau, self.alpha.value, self.beta, 3, self.num_basis.value, self.num_sensors)
            weights_pos, _ = dmp_pos_traj.train_using_individual_trajectory('position', trajectory_times, trajectory[:, :3])
            self.quat_canonical_goal_time=(self.truncated_trajectory_times[-1]-1.0)
            self.quat_alpha_phase = self.truncated_trajectory_times[-1]
            quat_alpha, quat_num_basis = self.alpha.value, self.num_basis.value
            quat_beta = quat_alpha / 4.0
            dmp_quat_traj = DMPTrajectory(self.tau, quat_alpha, quat_beta, 3, quat_num_basis, 1, add_min_jerk=False, 
                                          alpha_phase=self.quat_alpha_phase)
            # Quaternion DMPs right now use dt to get canonical variable.
            weights_quat, _ = dmp_quat_traj.train_using_individual_trajectory(
                'quaternion', trajectory_times, trajectory[:, 3:], dt=self.rollout_dt)
            y_pos, dy_pos, x_pos, _, _ = dmp_pos_traj.run_dmp_with_weights(weights_pos,
                                                                       np.zeros((3)),
                                                                       self.rollout_dt,
                                                                       traj_time=traj_time)

            q, qd, qdd, q_xlog = dmp_quat_traj.run_quaternion_dmp_with_weights(
                weights_quat, 
                trajectory[0, 3:], 
                trajectory[-1, 3:], 
                self.rollout_dt,
                traj_time=traj_time,
                quat_canonical_goal_time=self.quat_canonical_goal_time
            )   

            new_trajectory_times = np.arange(y_pos.shape[0]) * self.rollout_dt
            self.bot_plot_source.data = dict(x=new_trajectory_times.tolist(), 
                                             y0=np.array(y_pos[:,0]).tolist(), y1=np.array(y_pos[:,1]).tolist(), y2=np.array(y_pos[:,2]).tolist(), 
                                             y3=np.array(q[:,0]).tolist(), y4=np.array(q[:,1]).tolist(), y5=np.array(q[:,2]).tolist(), y6=np.array(q[:,3]).tolist())
        
            for i in range(7):
                self.p2.legend.items[i].update(label=value(self.cart_legend_labels[i]))

        else:
            trajectory_times = self.truncated_trajectory_times.reshape(-1, 1)
        
            trajectory = self.truncated_joint_trajectory - self.truncated_joint_trajectory[0,:]

            x = self.truncated_trajectory_times.tolist()
            ys1 = np.transpose(trajectory).tolist()
            
            self.top_plot_source.data = dict(x=x, y0=ys1[0], y1=ys1[1], y2=ys1[2], y3=ys1[3], y4=ys1[4], y5=ys1[5], y6=ys1[6])
            for i in range(7):
                self.p1.legend.items[i].update(label=value(self.joint_legend_labels[i]))

            

            dmp_traj = DMPTrajectory(self.tau, self.alpha.value, self.beta, self.num_dims, self.num_basis.value, self.num_sensors)
            weights, x_and_y = dmp_traj.train_using_individual_trajectory("joint", trajectory_times, trajectory)
            y, dy, _, _, _ = dmp_traj.run_dmp_with_weights(weights,
                                    np.zeros((dmp_traj.num_dims)),
                                    self.rollout_dt,
                                    traj_time=int(self.truncated_trajectory_times[-1] / self.rollout_dt))

            new_trajectory_times = np.arange(y.shape[0]) * self.rollout_dt
            self.bot_plot_source.data = dict(x=new_trajectory_times.tolist(), 
                                             y0=np.array(y[:,0]).tolist(), y1=np.array(y[:,1]).tolist(), y2=np.array(y[:,2]).tolist(), 
                                             y3=np.array(y[:,3]).tolist(), y4=np.array(y[:,4]).tolist(), y5=np.array(y[:,5]).tolist(), y6=np.array(y[:,6]).tolist())
            for i in range(7):
                self.p2.legend.items[i].update(label=value(self.joint_legend_labels[i]))

    def submit_callback(self):
        if self.dmp_type.active == 0:
            trajectory_times = self.truncated_trajectory_times.reshape(-1, 1)
        
            trajectory = self.truncated_cartesian_trajectory - self.truncated_cartesian_trajectory[0,:]

            traj_time = len(np.arange(0, self.truncated_trajectory_times[-1], self.rollout_dt))

            dmp_pos_traj = DMPTrajectory(self.tau, self.alpha.value, self.beta, 3, self.num_basis.value, self.num_sensors)
            weights_pos, _ = dmp_pos_traj.train_using_individual_trajectory('position', trajectory_times, trajectory[:, :3])
            self.quat_canonical_goal_time=(self.truncated_trajectory_times[-1]-1.0)
            self.quat_alpha_phase = self.truncated_trajectory_times[-1]
            quat_alpha, quat_num_basis = self.alpha.value, self.num_basis.value
            quat_beta = quat_alpha / 4.0
            dmp_quat_traj = DMPTrajectory(self.tau, quat_alpha, quat_beta, 3, quat_num_basis, 1, add_min_jerk=False, 
                                          alpha_phase=self.quat_alpha_phase)
            # Quaternion DMPs right now use dt to get canonical variable.
            weights_quat, _ = dmp_quat_traj.train_using_individual_trajectory(
                'quaternion', trajectory_times, trajectory[:, 3:], dt=self.rollout_dt)
            y_pos, dy_pos, x_pos, _, _ = dmp_pos_traj.run_dmp_with_weights(weights_pos,
                                                                       np.zeros((3)),
                                                                       self.rollout_dt,
                                                                       traj_time=traj_time)

            q, qd, qdd, q_xlog = dmp_quat_traj.run_quaternion_dmp_with_weights(
                weights_quat, 
                trajectory[0, 3:], 
                trajectory[-1, 3:], 
                self.rollout_dt,
                traj_time=traj_time,
                quat_canonical_goal_time=self.quat_canonical_goal_time
            )   

            new_trajectory_times = np.arange(y_pos.shape[0]) * self.rollout_dt
            self.bot_plot_source.data = dict(x=new_trajectory_times.tolist(), 
                                             y0=np.array(y_pos[:,0]).tolist(), y1=np.array(y_pos[:,1]).tolist(), y2=np.array(y_pos[:,2]).tolist(), 
                                             y3=np.array(q[:,0]).tolist(), y4=np.array(q[:,1]).tolist(), y5=np.array(q[:,2]).tolist(), y6=np.array(q[:,3]).tolist())

        else:
            trajectory_times = self.truncated_trajectory_times.reshape(-1, 1)
        
            trajectory = self.truncated_joint_trajectory - self.truncated_joint_trajectory[0,:]            

            dmp_traj = DMPTrajectory(self.tau, self.alpha.value, self.beta, self.num_dims, self.num_basis.value, self.num_sensors)
            weights, x_and_y = dmp_traj.train_using_individual_trajectory("joint", trajectory_times, trajectory)
            y, dy, _, _, _ = dmp_traj.run_dmp_with_weights(weights,
                                    np.zeros((dmp_traj.num_dims)),
                                    self.rollout_dt,
                                    traj_time=int(self.truncated_trajectory_times[-1] / self.rollout_dt))

            new_trajectory_times = np.arange(y.shape[0]) * self.rollout_dt
            self.bot_plot_source.data = dict(x=new_trajectory_times.tolist(), 
                                             y0=np.array(y[:,0]).tolist(), y1=np.array(y[:,1]).tolist(), y2=np.array(y[:,2]).tolist(), 
                                             y3=np.array(y[:,3]).tolist(), y4=np.array(y[:,4]).tolist(), y5=np.array(y[:,5]).tolist(), y6=np.array(y[:,6]).tolist())

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
