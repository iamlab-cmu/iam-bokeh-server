#!/usr/bin/env python

import rospy
from web_interface_msgs.msg import Request as webrequest
from bokeh_server_msgs.msg import Request as bokehrequest

def bokeh_server_test():
    web_pub = rospy.Publisher('/human_interface_request', webrequest, queue_size=1)
    bokeh_pub = rospy.Publisher('/bokeh_request', bokehrequest, queue_size=1)
    rospy.init_node('bokeh_server_test', anonymous=True)
    web_request_msg = webrequest()
    #web_request_msg.instruction_text = 'Label the image below by tapping on 4 extreme points. You can press on a point again to remove it. Then type in the name of the object and press submit. Press Done when you have finished labeling.'
    web_request_msg.instruction_text = 'Truncate the trajectory automatically or by using the sliders below. Then press continue to train the dmp using different parameters.'
    web_request_msg.display_type = 3
    web_pub.publish(web_request_msg)
    rospy.sleep(1)

    bokeh_request_msg = bokehrequest()
    bokeh_request_msg.display_type = 0
    bokeh_pub.publish(bokeh_request_msg)
    rospy.sleep(1)

if __name__ == '__main__':
    try:
        bokeh_server_test()
    except rospy.ROSInterruptException:
        pass