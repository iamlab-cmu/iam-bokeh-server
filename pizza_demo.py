#!/usr/bin/env python

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column, Button, TextInput, PointDrawTool

from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.layouts import column, row

from threading import Thread
import numpy as np
from std_msgs.msg import String

from bokeh_server_msgs.msg import *

class PizzaDemo:

    def __init__(self, doc, pub):
        self.doc = doc
        self.pub = pub

        self.query_type = 0

        self.demonstration_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.correction_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.p1 = figure(title='',
                   width=500, height=500,
                   x_axis_type=None, y_axis_type=None, 
                   x_range=(0, 10), y_range=(0, 10))

        self.p2 = figure(title='',
                   width=500, height=500,
                   x_axis_type=None, y_axis_type=None, 
                   x_range=(0, 10), y_range=(0, 10))

        self.p3 = figure(title='',
                   tools='tap', width=500, height=500,
                   x_axis_type=None, y_axis_type=None, 
                   x_range=(0, 10), y_range=(0, 10))

        # add a button widget and configure with the call back
        self.submit_button = Button(label="Submit")
        self.submit_button.on_click(self.submit_callback)


        self.pizza1_button = Button(label="Pizza1")
        self.pizza1_button.on_click(self.pizza1_callback)

        self.pizza2_button = Button(label="Pizza2")
        self.pizza2_button.on_click(self.pizza2_callback)

        self.good_button = Button(label="Good")
        self.good_button.on_click(self.good_callback)

        self.bad_button = Button(label="Bad")
        self.bad_button.on_click(self.bad_callback)

        self.demonstration_coord_list=[]
        self.correction_coord_list=[]

    #add a dot where the click happened
    def tap_callback(self, event):
        coords=(event.x,event.y)

        if len(self.demonstration_coord_list) == 0:
            self.demonstration_coord_list.append(coords)
        else:
            self.demonstration_coord_list = [coords]

        self.demonstration_source.data = dict(x=[i[0] for i in self.demonstration_coord_list], y=[i[1] for i in self.demonstration_coord_list])

    def submit_callback(self):
        self.doc.clear()

        response_msg = Response()
        response_msg.query_type = self.query_type
        if self.query_type == 2:
            response_msg.query_point.x = self.demonstration_source.data['x'][0] * self.pizza_diameter / 10
            response_msg.query_point.y = self.demonstration_source.data['y'][0] * self.pizza_diameter / 10
        elif self.query_type == 3:
            response_msg.query_point.x = self.correction_source.data['x'][0] * self.pizza_diameter / 10
            response_msg.query_point.y = self.correction_source.data['y'][0] * self.pizza_diameter / 10

        self.pub.publish(response_msg)


    def pizza1_callback(self):
        self.doc.clear()

        response_msg = Response()
        response_msg.query_type = self.query_type
        response_msg.query_response = 0
        
        self.pub.publish(response_msg)

    def pizza2_callback(self):
        self.doc.clear()

        response_msg = Response()
        response_msg.query_type = self.query_type
        response_msg.query_response = 1

        self.pub.publish(response_msg)

    def good_callback(self):
        self.doc.clear()

        response_msg = Response()
        response_msg.query_type = self.query_type
        response_msg.query_response = 1
        
        self.pub.publish(response_msg)

    def bad_callback(self):
        self.doc.clear()

        response_msg = Response()
        response_msg.query_type = self.query_type
        response_msg.query_response = 0

        self.pub.publish(response_msg)


    def handle_pizza_demo_request(self, request):
        print('Received Pizza Request')

        self.query_type = request.pizza.query_type
        self.pizza_diameter = request.pizza.pizza_diameter
        crust_thickness = request.pizza.crust_thickness
        topping_diameter = request.pizza.topping_diameter

        cheese_radius = 5 * (1-((crust_thickness*2)/ self.pizza_diameter))
        topping_radius = 5 * topping_diameter / self.pizza_diameter

        pizza1_x = np.array(request.pizza.pizza_topping_positions_1_x) * 10 / self.pizza_diameter + 5
        pizza1_y = np.array(request.pizza.pizza_topping_positions_1_y) * 10 / self.pizza_diameter + 5

        # Query Type 0 is Preference
        if self.query_type == 0:

            print('Preference')
            pizza2_x = np.array(request.pizza.pizza_topping_positions_2_x) * 10 / self.pizza_diameter + 5
            pizza2_y = np.array(request.pizza.pizza_topping_positions_2_y) * 10 / self.pizza_diameter + 5

            pizza1_source = ColumnDataSource(data=dict(x=pizza1_x[:-1], y=pizza1_y[:-1]))
            pizza2_source = ColumnDataSource(data=dict(x=pizza2_x[:-1], y=pizza2_y[:-1]))

            self.p1.circle(x=[5], y=[5], radius=5, fill_color='peru')
            self.p1.circle(x=[5], y=[5], radius=cheese_radius, fill_color='gold')
            self.p1.circle(source=pizza1_source, x='x', y='y', radius=topping_radius, fill_color='red')

            self.p1.circle(x=[pizza1_x[-1]], y=[pizza1_y[-1]], radius=topping_radius, fill_color='green') 

            self.p2.circle(x=[5], y=[5], radius=5, fill_color='peru')
            self.p2.circle(x=[5], y=[5], radius=cheese_radius, fill_color='gold')
            self.p2.circle(source=pizza2_source, x='x', y='y', radius=topping_radius, fill_color='red') 

            self.p2.circle(x=[pizza2_x[-1]], y=[pizza2_y[-1]], radius=topping_radius, fill_color='green') 

            self.page_layout=column(row(self.p1, self.p2), row(self.pizza1_button, self.pizza2_button))
        
        # Query Type 1 is Binary Feedback
        elif self.query_type == 1:
            print('Binary')
            pizza1_source = ColumnDataSource(data=dict(x=pizza1_x[:-1], y=pizza1_y[:-1]))

            self.p1.circle(x=[5], y=[5], radius=5, fill_color='peru')
            self.p1.circle(x=[5], y=[5], radius=cheese_radius, fill_color='gold')
            self.p1.circle(source=pizza1_source, x='x', y='y', radius=topping_radius, fill_color='red') 

            self.p1.circle(x=[pizza1_x[-1]], y=[pizza1_y[-1]], radius=topping_radius, fill_color='green') 

            self.page_layout=column(self.p1, row(self.good_button, self.bad_button))

        # Query Type 2 is Demonstration
        elif self.query_type == 2:
            print('Demo')
            pizza1_source = ColumnDataSource(data=dict(x=pizza1_x, y=pizza1_y))

            self.p3.circle(x=[5], y=[5], radius=5, fill_color='peru')
            self.p3.circle(x=[5], y=[5], radius=cheese_radius, fill_color='gold')
            self.p3.circle(source=pizza1_source, x='x', y='y', radius=topping_radius, fill_color='red') 

            self.p3.circle(source=self.demonstration_source, x='x', y='y', radius=topping_radius, fill_color='green') 

            self.p3.on_event(Tap, self.tap_callback)

            self.page_layout=column(self.p3, self.submit_button)

        # Query Type 3 is Correction
        elif self.query_type == 3:
            print('Correction')
            pizza1_source = ColumnDataSource(data=dict(x=pizza1_x[:-1], y=pizza1_y[:-1]))

            self.p1.circle(x=[5], y=[5], radius=5, fill_color='peru')
            self.p1.circle(x=[5], y=[5], radius=cheese_radius, fill_color='gold')
            self.p1.circle(source=pizza1_source, x='x', y='y', radius=topping_radius, fill_color='red') 

            self.correction_source.data = dict(x=[pizza1_x[-1]], y=[pizza1_y[-1]])

            c1 = self.p1.circle(source=self.correction_source, x='x', y='y', radius=topping_radius, fill_color='green') 

            tool = PointDrawTool(add=False, renderers=[c1])

            self.page_layout=column(self.p1, self.submit_button)

        self.doc.add_root(self.page_layout)
