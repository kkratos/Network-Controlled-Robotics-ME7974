import numpy as np
import pybullet as p
import itertools

class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []
        

    def reset(self):
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        return self.messages_received
        
    def send_message(self, robot_id, message):
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        return self.neighbors
    
    def compute_controller(self):
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """
        
        # here we implement an example for a consensus algorithm
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()
        
        #send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, pos)
        
        p_des = [[0.5, -1], [0.5, 1], [0.5, 0], [1.5, 0], [2.5, -1], [2.5, 1]]
        k_f = 10
        K_t = 5
        
        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position])
        dx = 0.
        dy = 0.
        L = np.zeros((1, 6))
        Px_des = np.zeros((6, 1))
        Py_des = np.zeros((6, 1))
       
        if messages:
            for m in messages:
                
                L[0, m[0]] = -1
                
                Px_des[m[0]] = p_des[m[0]][0] - m[1][0]
                Py_des[m[0]] = p_des[m[0]][1] - m[1][1]
            
            L[0, self.id] = len(messages)
            Px_des[self.id] = p_des[self.id][0] - pos[0]
            Py_des[self.id] = p_des[self.id][1] - pos[1]
            
            
        dx = k_f * np.matmul(L, Px_des) + K_t * (p_des[self.id][0] - pos[0])
        dy = k_f * np.matmul(L, Py_des) + K_t * (p_des[self.id][1] - pos[1])
            # integrate
        des_pos_x = pos[0] + self.dt * dx
        des_pos_y = pos[1] + self.dt * dy
        
        #compute velocity change for the wheels
        vel_norm = np.linalg.norm([dx, dy]) #norm of desired velocity
        if vel_norm < 0.01:
            vel_norm = 0.01
        des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
        right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
        left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
        self.set_wheel_velocity([left_wheel, right_wheel])
        

    
       
