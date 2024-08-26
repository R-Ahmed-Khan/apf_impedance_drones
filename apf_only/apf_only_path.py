import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from apf import APF_Improved, Vector2d

class APF_ONLY:
    def __init__(self):
        pass

    def simulate(self, num_drones=2, start_pos=(-3, 1), goal_pose=(6, -1), obstacles_pos=[[0.305, 0.528], [-0.305, -0.528]], \
                 r_apf_list=[0.5, 0.5], d_sep=0.45, step=0.1, plot=True, trail_drones=[1],rotation=0):
        '''
        Each drone uses the APF method independently for navigation.
        Drones start with a specific separation and orientation.
        '''
        apf_instances = []
        drone_paths = []
        goal_threshold = 0.2

        # Calculate initial positions with separation and orientation
        initial_positions = []
        for i in range(num_drones):
            angle = (-2 * np.pi * i) / num_drones - rotation/5  # Distribute drones in a circle around the start position
            offset = np.array([d_sep * np.cos(angle), d_sep * np.sin(angle)])
            drone_start_pos = np.array(start_pos) + offset
            initial_positions.append(drone_start_pos)
        
        # Calculate individual goal positions based on the single goal position
        goal_poses = []
        for i in range(num_drones):
            angle = (-2 * np.pi * i) / num_drones - rotation/5
            offset = np.array([d_sep * np.cos(angle), d_sep * np.sin(angle)])
            drone_goal_pos = np.array(goal_pose) + offset
            goal_poses.append(drone_goal_pos)
        
        # Initialize APF instances for each drone with their respective start positions
        for i in range(num_drones):
            apf = APF_Improved(start=initial_positions[i], goal=goal_poses[i], obstacles=obstacles_pos, k_att=1.0, k_rep=0.8, \
                               rr_list=r_apf_list, step_size=step, max_iters=1500, goal_threshold=0.2, is_plot=False)
            apf.path_plan()
            drone_paths.append(np.array(apf.path))
            apf_instances.append(apf)
        
        drone_poses = {}
        for i in range(num_drones):
            drone_poses[f'Drone_{i+1}'] = []

        if plot:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xticks(np.arange(-3.5, 7, 1))
            ax.set_yticks(np.arange(-3.5, 7, 1))
            ax.grid(True)
        
        drone_trails_x = [[] for _ in range(num_drones)]
        drone_trails_y = [[] for _ in range(num_drones)]

        def update(frame):
            ax.clear()
            ax.grid(True)
            ax.set_xlim(-3.5, 2.5)
            ax.set_ylim(-1.5, 3.5)
            ax.set_xlabel('X (m)', fontsize=14)
            ax.set_ylabel('Y (m)', fontsize=14)
            ax.set_aspect('equal')
            
            all_drones_at_goal = True

            for i, path in enumerate(drone_paths):
                if frame < len(path):
                    drone_pose = path[frame]
                else:
                    drone_pose = path[-1] # Stays at last position

                drone_poses[f'Drone_{i+1}'].append(drone_pose)
                ax.scatter(drone_pose[0], drone_pose[1], c='b')
                ax.text(drone_pose[0], drone_pose[1], f'Drone {i+1}', fontsize=12)

                if i + 1 in trail_drones:
                    drone_trails_x[i].append(drone_pose[0])
                    drone_trails_y[i].append(drone_pose[1])
                    # Plot only one label per drone for the trail
                    ax.plot(drone_trails_x[i], drone_trails_y[i], linestyle='--', color='black')
                
                # Check if the drone has reached the goal
                if np.linalg.norm(drone_pose - goal_poses[i]) > goal_threshold:
                    all_drones_at_goal = False

            ax.plot(start_pos[0], start_pos[1], '*r', label='Start Position', markersize=10)
            ax.plot(goal_pose[0], goal_pose[1], '*g', label='Goal Position', markersize=10)

            # Plot obstacles
            for i, obstacle in enumerate(obstacles_pos):
                rr_apf = r_apf_list[i]
                obstacle_circle = plt.Circle(obstacle, rr_apf, color='darkblue', alpha=0.7)
                ax.add_patch(obstacle_circle)
                ax.plot(obstacle[0], obstacle[1], 'xk')
            
            ax.plot([], [],'k--', label='APF Trajectory') 
            # Legend for the APF obstacle region
            ax.plot([], [], 'darkblue', alpha=0.7, label='APF Repulsion Field of Obstacle', linewidth=10)
            plt.legend(loc='lower left', fontsize='large')
            
            # Stop the animation if all drones have reached the goal
            if all_drones_at_goal:
                anim.event_source.stop()

        if plot:
            # Update frames to the length of the longest path
            anim = FuncAnimation(fig, update, frames=range(max(len(path) for path in drone_paths)), repeat=False)
            plt.show()

        return drone_poses

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
#     drone_poses_dict = instance.simulate(num_drones=4, start_pos=(0, 0), goal_pose=(5, 5), obstacles_pos=[[2, 2.9], [2.9, 2]], r_apf_list=[0.5, 0.5], d_sep=0.5, step=0.1, plot=True)
#     plot_drone_poses(drone_poses_dict)

# if __name__ == "__main__":
#     main()
