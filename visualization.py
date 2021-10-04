import numpy as np
import matplotlib.pyplot as plt

def visualize_cartesian_position_trajectories(trajectory_times, trajectory, labels=['x','y','z'], fig_num=1, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(3):
        ax.plot(trajectory_times, trajectory[:,i], label=labels[i])
    ax.legend()
    ax.set_title('Cartesian Position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    
def visualize_cartesian_orientation_trajectories(trajectory_times, trajectory, labels=['rx','ry','rz'], fig_num=2, ax=None):
        
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(3):
        ax.plot(trajectory_times, trajectory[:,i+3], label=labels[i])
    ax.legend()
    ax.set_title('Cartesian Orientation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    
def visualize_cartesian_trajectories(trajectory_times, trajectory, labels=['x','y','z','rx','ry','rz'], fig_num=3, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(6):
        if labels is None:
            ax.plot(trajectory_times, trajectory[:,i])
        else:
            ax.plot(trajectory_times, trajectory[:,i], label=labels[i])
    if labels is not None:
        ax.legend()
    ax.set_title('Cartesian Pose')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m) and Angle (rad)')

def visualize_relative_cartesian_position_trajectories(trajectory_times, trajectory, labels=['x','y','z'], fig_num=4, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(3):
        ax.plot(trajectory_times, trajectory[:,i] - trajectory[0,i], label=labels[i])
    ax.legend()
    ax.set_title('Relative Cartesian Position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')

def visualize_relative_cartesian_orientation_trajectories(trajectory_times, trajectory, labels=['rx','ry','rz'], fig_num=5, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(3):
        ax.plot(trajectory_times, trajectory[:,i+3] - trajectory[0,i+3], label=labels[i])
    ax.legend()
    ax.set_title('Relative Cartesian Orientation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    
def visualize_relative_cartesian_trajectories(trajectory_times, trajectory, labels=['x','y','z','rx','ry','rz'], fig_num=6, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(6):
        if labels is None:
            ax.plot(trajectory_times, trajectory[:,i] - trajectory[0,i])
        else:
            ax.plot(trajectory_times, trajectory[:,i] - trajectory[0,i], label=labels[i])
    if labels is not None:
        ax.legend()
    ax.set_title('Relative Cartesian Pose')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m) and Angle (rad)')

def visualize_joint_trajectories(trajectory_times, trajectory, labels=['1','2','3','4','5','6','7'], fig_num=7, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(len(labels)):
        ax.plot(trajectory_times, trajectory[:,i], label=labels[i])
    ax.legend()
    ax.set_title('Joint Position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    
def visualize_relative_joint_trajectories(trajectory_times, trajectory, labels=['1','2','3','4','5','6','7'], fig_num=8, ax=None):
    
    plt.figure(fig_num)
    ax = plt.gca() if ax is None else ax
    for i in range(len(labels)):
        ax.plot(trajectory_times, trajectory[:,i] - trajectory[0,i], label=labels[i])
    ax.legend()
    ax.set_title('Relative Joint Position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
