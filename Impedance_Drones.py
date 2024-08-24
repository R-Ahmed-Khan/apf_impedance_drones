import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
import time
from APF import APF_Improved, Vector2d
from matplotlib.patches import Patch

class APF_IMP:
    def __init__(self):
        pass
    
    def MassSpringDamper(self, state, t, F):
        x = state[0]
        xd = state[1]
        m = 1
        b = 0.2
        k = 0.2
        # m = 1.9
        # b = 12.6
        # k = 20.88
        xdd = -b/m * xd + (-k/m) * x + F/m
        return [xd, xdd]

    def impedance(self, leader_vel, imp_pose_prev, imp_vel_prev, time_prev):
        F_coeff = 4  
        time_step = time.time() - time_prev
        time_prev = time.time()
        t = [0., time_step]
        F = 0.01 * F_coeff * leader_vel
        num_drones = len(leader_vel)

        imp_pose = np.zeros_like(imp_pose_prev)
        imp_vel = np.zeros_like(imp_vel_prev)

        for s in range(num_drones):
            state0 = [imp_pose_prev[s], imp_vel_prev[s]]
            state = self.MassSpringDamper(state0, t, F[s])
            imp_pose[s] = state[0]*time_step
            imp_vel[s] = state[1]*time_step

        return imp_pose, imp_vel, time_prev
    
    def impedance_obs(self, curr_drone_pos, obstacle_center, rr_imp, imp_vel_prev, time_prev, leader_pos):
        # F_coeff = 0.40
        # dir_to_center = obstacle_center - curr_drone_pos
        # distance_to_obstacle = np.linalg.norm(dir_to_center)
        
        # if distance_to_obstacle < rr_imp:
        #     dir_to_center /= distance_to_obstacle  # Normalize direction vector

        #     # Calculate the deflection force away from the obstacle
        #     deflection_distance = F_coeff * (rr_imp - distance_to_obstacle)  
        #     avoidance_movement = -deflection_distance * dir_to_center

        #     # Calculate the direction towards the leader
        #     dir_to_leader = leader_pos - curr_drone_pos
        #     dir_to_leader /= np.linalg.norm(dir_to_leader)  # Normalize direction vector

        #     # Combine the avoidance movement with movement towards the leader
        #     movement_towards_leader = 0.1 * dir_to_leader  # 0.1 is a small weight towards the leader
        #     curr_drone_pos += avoidance_movement + movement_towards_leader
        # else:
        #     # If outside the rr_imp region, move towards the leader only
        #     dir_to_leader = leader_pos - curr_drone_pos
        #     curr_drone_pos += 0.25 * dir_to_leader / np.linalg.norm(dir_to_leader)
        F_coeff = 0.45
        dir_to_center = obstacle_center - curr_drone_pos
        dir_to_center /= np.linalg.norm(dir_to_center)  # Normalize direction vector
        deflection_distance = F_coeff * rr_imp  
        curr_drone_pos -= deflection_distance * dir_to_center  
        
        return curr_drone_pos 

    def simulate(self, num_drones=2, start_pos=(-3,1), goal_pose=(6,-1), obstacles_pos=[[0.305,0.528],[-0.305, -0.528]], \
                  r_apf_list=[0.5, 0.5], r_imp_list=[0.18, 0.18], d_sep=0.5, step=0.1, plot=True, trail_drones=[1], rotation=0):
        '''
        change APF parameters in function call
        k_att => attraction force
        k_rep => repulsion force
        '''
        apf = APF_Improved(start=start_pos, goal=goal_pose, obstacles=obstacles_pos, k_att=1.0, k_rep=0.8, \
                            rr_list=r_apf_list, step_size=step, max_iters=1500, goal_threshold=0.2, is_plot=False)
        start1 = apf.start
        start1_str = str(start1)
        goal1 = apf.goal
        goal1_str = str(goal1)

        # Split the string by commas and extract deltaX and deltaY
        attributes1 = start1_str.split(',')
        attributes2 = goal1_str.split(',')
        
        deltaX = float(attributes1[0].split(':')[1])  # Extracting deltaX
        deltaY = float(attributes1[1].split(':')[1])  # Extracting deltaY
        start = [deltaX,deltaY]
       
        deltaXg = float(attributes2[0].split(':')[1])  # Extracting deltaX
        deltaYg = float(attributes2[1].split(':')[1])  # Extracting deltaY
        goal = [deltaXg,deltaYg]

        apf.path_plan()
        path = apf.path
        leader_pose = np.array(path)  # Start from the first point of the trajectory
        zeros_column = np.zeros((leader_pose.shape[0], 1))
        leader_pose_temp = np.hstack((leader_pose,zeros_column))
        leader_vel_temp = np.zeros_like(leader_pose_temp)
        time_step = 0.01
        leader_vel_temp[1:] = (leader_pose_temp[1:] - leader_pose_temp[:-1]) / time_step

        leader_poses = []
        drone_poses = {}
        for i in range(num_drones):
            drone_poses[f'Drone_{i+1}'] = [] 

        L_distance = d_sep
        # rr_imp = r_imp
        drone_labels = [f'Drone {i+1}' for i in range(num_drones)]
        imp_pose = np.zeros(shape=(num_drones, 3))
        imp_vel = np.zeros(shape=(num_drones, 3))
        imp_time_prev = np.zeros(num_drones)
        imp_pose_prev = np.zeros(shape=(num_drones, 3))
        imp_vel_prev = np.zeros(shape=(num_drones, 3))
        count = 0
        if plot:
            fig, ax = plt.subplots(figsize=(7, 7))
            # Plot grid
            ax.set_xticks(np.arange(0, 6, 1))
            ax.set_yticks(np.arange(0, 6, 1))
            # Plot starting and ending goal points
            ax.scatter(0, 0, c='g', marker='o', label='Start')
            ax.scatter(5, 5, c='b', marker='o', label='End')

        # Extracting obstacles center location from vector
        obstacles = apf.obstacles
        self.obstacle_circles = []
        count = 1
        for i, obstacle in enumerate(obstacles):
            obs_cent_str = str(obstacle)
            obs_cent = obs_cent_str.split(',')
            obs_x = float(obs_cent[0].split(':')[1])
            obs_y = float(obs_cent[1].split(':')[1])
            
            # Use individual radii
            rr_apf = r_apf_list[i]
            rr_imp = r_imp_list[i]
            
            obstacle_circle = plt.Circle((obs_x, obs_y), rr_apf, color='darkblue', alpha=0.7)
            self.obstacle_circles.append(obstacle_circle)
            obstacle_solid_circle = plt.Circle((obs_x, obs_y), rr_imp, color='lightgreen', alpha=0.69)
            self.obstacle_circles.append(obstacle_solid_circle)
            count = count + 1
        leader_trail_x = []
        leader_trail_y = []
        drone_trail_x = {f'Drone_{i+1}': [] for i in range(num_drones)}
        drone_trail_y = {f'Drone_{i+1}': [] for i in range(num_drones)}

        drone_poses_dict = {f'Drone_{i+1}': [] for i in range(num_drones)}

        def update(frame):
            nonlocal count
            count += 1
            if plot:
                ax.clear()
                ax.grid(True)
                ax.set_xlim(-3.5, 2.5)
                ax.set_ylim(-1.5, 3.5)
                ax.set_xlabel('X (m)', fontsize=14)
                ax.set_ylabel('Y (m)', fontsize=14)
                ax.set_aspect('equal')  

            for s in range(num_drones):
                leader_vel = leader_vel_temp[frame]
                imp_pose[s, :], imp_vel[s, :], imp_time_prev[s] = self.impedance(leader_vel, imp_pose_prev[s, :], imp_vel_prev[s, :], imp_time_prev[s])
                imp_pose_prev[s, :] = imp_pose[s, :]
                imp_vel_prev[s, :] = imp_vel[s, :]
            poses = []
            for s in range(num_drones):
                leader_pose = leader_pose_temp[frame]
                angle = (2*np.pi*s)/num_drones + rotation
                drone_pose = leader_pose + np.array([-L_distance * np.cos(angle), L_distance * np.sin(angle), 0]) + 0.2 * imp_pose[s, :] 
                obstacle_avoided = False
                # Check if drone is within any obstacle circle

                # Iterate over each obstacle with index i
                for i, obstacle_circle in enumerate(self.obstacle_circles):
                    obstacle_center = obstacle_circle.center
                    rr_imp = r_imp_list[i // 2]  # Assuming each obstacle has an associated radius
                    
                    if np.linalg.norm(drone_pose[:2] - obstacle_center) < (rr_imp + 0.25):
                        obstacle_avoided = True
                        # Move drone away from obstacle                    
                        drone_pose[:2] = self.impedance_obs(drone_pose[:2], obstacle_center, rr_imp, imp_vel_prev, 1, leader_pose[:2]) 
                        
                        if plot:
                            center = obstacle_center
                            radius = rr_imp 
                            angle = np.linspace(0, 2*np.pi, 100)
                            x_circle = center[0] + radius * np.cos(angle)
                            y_circle = center[1] + radius * np.sin(angle)
                            ax.plot(x_circle, y_circle, 'k--')  # Plot circular trajectory
                            ax.plot([drone_pose[0], center[0]], [drone_pose[1], center[1]], 'g--') 

                if not obstacle_avoided:
                    if plot:
                        ax.plot([drone_pose[0], leader_pose[0]], [drone_pose[1], leader_pose[1]], 'g--')

                poses.append(drone_pose)
                drone_poses_dict[f'Drone_{s+1}'].append(drone_pose)

                if s+1 in trail_drones:
                    drone_trail_x[f'Drone_{s+1}'].append(drone_pose[0])
                    drone_trail_y[f'Drone_{s+1}'].append(drone_pose[1])

            # Plot trails of the leader drone
            leader_trail_x.append(leader_pose[0])
            leader_trail_y.append(leader_pose[1])
            if plot:
                ax.plot(leader_trail_x, leader_trail_y, 'r--', label='APF Trajectory')
                ax.scatter(leader_pose[0], leader_pose[1], c='r', label='Virtual Leader')
                for i, pose in enumerate(poses):
                    ax.scatter(pose[0], pose[1], c='b')
                    ax.text(pose[0], pose[1], drone_labels[i])

                for s in trail_drones:
                    ax.plot(drone_trail_x[f'Drone_{s}'], drone_trail_y[f'Drone_{s}'], 'k--')

            # Plot start and goal points on the trajectory

                ax.plot(start[0], start[1], '*r', label='Start Position', markersize=10)
                ax.plot(goal[0], goal[1], '*g', label='Goal Position', markersize=10)    

                for i, obstacle_circle in enumerate(self.obstacle_circles):
                    ax.add_patch(obstacle_circle)
                    center_x, center_y = obstacle_circle.center
                    ax.plot(center_x, center_y, 'xk')
                ax.plot([], [],'k--', label='Drone Trail') 
                ax.plot([], [],'darkblue', alpha=0.7, label='APF Repulsion Field of Obstacle', linewidth=10)
                ax.plot([],[], 'lightgreen', alpha=0.7, label='Local Deflection Field of Obstacle', linewidth=10)
                plt.legend(loc='lower left', fontsize='large') 
                
            return []
        # print('-------------------------')
        # print(drone_poses_dict)

        if plot:
            anim = FuncAnimation(fig, update, frames=range(leader_pose_temp.shape[0]), blit=True, repeat=False)
            plt.show()
        else:
            for frame in range(leader_pose_temp.shape[0]):
                update(frame)
        # print(drone_poses_dict)
        return drone_poses_dict, leader_pose

def plot_drone_poses(drone_poses_dict):
    num_drones = len(drone_poses_dict)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    for drone, poses in drone_poses_dict.items():
        x_pos = [pose[0] for pose in poses]
        y_pos = [pose[1] for pose in poses]

        axes[0].plot(x_pos, label=drone)
        axes[1].plot(y_pos, label=drone)

    axes[0].set_title('Drone Poses - X Coordinate')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('X Coordinate')
    axes[0].legend(fontsize='large')

    axes[1].set_title('Drone Poses - Y Coordinate')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].legend(fontsize='large')

    plt.tight_layout()
    plt.show()

# def main():
#     instance = APF_IMP()
#     drone_poses_dict = instance.simulate(num_drones=4,start_pos=(0,0),goal_pose=(5,5),obstacles_pos=[[2, 2.9],[2.9,2]],r_apf=0.5,r_imp=0.18,d_sep=0.5,plot=True)
#     # for drone, poses in drone_poses_dict.items():
#     #     print(f"{drone}: {poses}") 

# if __name__ == "__main__":
#     main()