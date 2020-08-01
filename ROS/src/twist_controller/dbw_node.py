#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from lowpass import LowPassFilter

#DEBUG JWD
#from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane
#ENDEBUG

import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity     = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit       = rospy.get_param('~decel_limit', -5)
        accel_limit       = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base        = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio       = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel     = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle   = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        parms = {
            'wheel_base'      : wheel_base,
            'steer_ratio'     : steer_ratio,
            'min_velocity'    : 0.,
            'max_lat_accel'   : max_lat_accel,
            'max_steer_angle' : max_steer_angle,
            'decel_limit'     : decel_limit,
            'accel_limit'     : accel_limit,
            'deadband'        : self.brake_deadband
        }
        self.controller = Controller(**parms)
 
        self.current_command  = None
        self.current_velocity = None
        self.dbw_enabled      = True
        
        rospy.Subscriber('/twist_cmd', TwistStamped, self.callback_twist_cmd)
        rospy.Subscriber('/current_velocity', TwistStamped, self.callback_current_velocity)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.callback_dbw_enabled)
        
        self.loop()

    # Get predicted throttle, brake, and steering using `controller` object
    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            if (self.current_command is not None) and (self.current_velocity is not None):
                # get the current velocity, target velocity, and target angle and pass into control
                linear_target  = self.current_command.twist.linear.x;
                angular_target = self.current_command.twist.angular.z;
                linear_current = self.current_velocity.twist.linear.x;
                angular_current = self.current_velocity.twist.angular.z;
                #rospy.loginfo("linearc: %.2f, lineart: %.2f, angularc: %.2f, angulart: %.2f", linear_current, linear_target, angular_current, angular_target)
                throttle, brake, steering = self.controller.control(linear_target, angular_target, linear_current)
                
                # publish the control commands if dbw is enabled
                if self.dbw_enabled is True:
                    #rospy.loginfo("v: %.2f, vtarg: %.2f, thr: %.2f, b: %.2f, s: %.2f", linear_current, linear_target, throttle, brake, steering)
                    self.publish(throttle, brake, steering)
                else:
                    self.controller.reset()
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        if throttle > 0.0:
            tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)
        
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        
        bcmd.pedal_cmd_type = BrakeCmd.CMD_PERCENT
        if self.brake_deadband > 0.1:
            brake *= 1000  # multiplier for sim
        bcmd.pedal_cmd = brake
        
        # alternate torque braking
        #bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        #if brake > 0. and throttle == 0.:
            #bcmd.pedal_cmd = brake * self.vehicle_mass * self.wheel_radius
        #    bcmd.pedal_cmd = BrakeCmd.TORQUE_MAX * brake    # 3412.
        self.brake_pub.publish(bcmd)

    def callback_twist_cmd(self, msg):
        self.current_command = msg

    def callback_current_velocity(self, msg):
        self.current_velocity = msg

    def callback_dbw_enabled(self, msg):
        rospy.loginfo("dbw_enabled: %s", msg)
        self.dbw_enabled = msg.data

if __name__ == '__main__':
    DBWNode()
