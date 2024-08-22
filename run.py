from Impedance_Drones import*
from APF import*
from pprint import pprint

def get_drone_poses():
    '''
    num_drones = 4 (default)
    start_pos = initial point of APF trajectory
    goal_pos = goal point of APF trajectory
    obstacles_pos = position of obstacles, add as much obstacles required in the given format e.g. [[2,2.9],[2.9,2],[obs_new_x,obs_new_y]]
    r_apf_list = radius list of APF repulsion field around obstacles
    r_imp_list = radius list of repulsive force applied to any drone in vicinty around obstacles
    d_sep = seperation b/w drones 
    step = step size of trajectory
    '''
    instance = APF_IMP()
    #obstacles_pos = [[-0.36, -0.64],[0.38, 0.69], [2, -0.8],[-2.02, 0.48]]  # case 1
    #drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-4.0,1.5), goal_pose=(3.42, -1.55), obstacles_pos=obstacles_pos, r_apf=0.24, r_imp=0.2, d_sep=0.45, step=0.083, plot=True)
    
    # obstacles_pos = [[-0.004, 0.96],[0.076, -0.826], [1.9, -0.19],[-1.86, 0.57]]  # case 2
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-4.0,1.5), goal_pose=(3.42, -1.05), obstacles_pos=obstacles_pos, r_apf=0.24, r_imp=0.2, d_sep=0.45, step=0.087, plot=True)
    
    ## case #1 -- gates only 
    # obstacles_pos = [[0.53, 1.03],[-0.53, -0.03]]  
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,3.0), goal_pose=(3.0, -2.0), obstacles_pos=obstacles_pos, \
    #                                            r_apf_list=[0.32,0.32], r_imp_list=[0.26,0.26], \
    #                                             d_sep=0.45, step=0.05, plot=True) 

    # case #2 -- obstacles only 
    # obstacles_pos = [[-1.0,1.0],[1.0, 0.0]]  
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,3.0), goal_pose=(3.0, -2.0), obstacles_pos=obstacles_pos, \
    #                                           r_apf_list=[0.32,0.32], r_imp_list=[0.22,0.22], \
    #                                            d_sep=0.45, step=0.03, plot=True)       

    ## case #3 -- obstacles only 
    # obstacles_pos = [[0.53, 1.03],[-0.53, -0.03], [-1.0,1.0],[1.0, 0.0]]  
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,3.0), goal_pose=(3.0, -2.0), obstacles_pos=obstacles_pos, \
    #                                             r_apf_list=[0.32,0.32,0.32,0.32], r_imp_list=[0.26,0.26,0.22,0.22], \
    #                                             d_sep=0.45, step=0.03, plot=True)       
    
    ## case #4 -- obstacles only -- more cluttered 
    obstacles_pos = [[-2,1.5],[-0.5,0.5],[1,-1],[2,-1.0],[2,1],[1,1],[-1.5,-1.5]]  
    drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,3.0), goal_pose=(3.0, -2.0), obstacles_pos=obstacles_pos, \
                                               r_apf_list=[0.6,0.6,0.3,0.25,0.40,0.25,0.6], r_imp_list=[0.40,0.40,0.25,0.20,0.30,0.15,0.40], \
                                                d_sep=0.45, step=0.03, plot=True)
    
    # pprint(drone_poses_dict)
    # plot_drone_poses(drone_poses_dict)
    
    return drone_poses_dict, path, obstacles_pos

def main():

    drone_poses_dict, path, obstacles_pos= get_drone_poses()
    print(path.shape[0])
    # print('shape = ', drone_poses_dict.shape[0])
    # Save drone poses dictionary to a file using pickle
    preprocessed_drone_poses = {}
    for drone, poses in drone_poses_dict.items():
        preprocessed_poses = [] 
        for pose in poses:
            preprocessed_poses.append(' '.join(map(str, pose)))
        preprocessed_drone_poses[drone] = '\n'.join(preprocessed_poses)

    # Save preprocessed drone poses dictionary to a file
    with open('drone_poses.txt', 'w') as f:
        for drone, poses in preprocessed_drone_poses.items():
            f.write(f"{drone}:\n{poses}\n\n")
    APF_norm = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))]
    APF_Trajectory = np.sum(APF_norm)
    print('Trajectory APF = ', APF_Trajectory)

    drone_norms = {}
    for drone, poses in drone_poses_dict.items():
        distances = [np.linalg.norm(poses[i] - poses[i-1]) for i in range(1, len(poses))]
        total_distance = np.sum(distances)
        drone_norms[drone] = total_distance
    print('Trajectory length of Drones = ',drone_norms)
    
    num_obstacles = len(obstacles_pos)
    min_distances_to_obstacles = [float('inf')] * num_obstacles
    min_poses_to_obstacles = [None] * num_obstacles

    for drone, poses in drone_poses_dict.items():
        for pose in poses:
            for i, obstacle_pos in enumerate(obstacles_pos):
                d_to_obstacle = np.linalg.norm(np.array(pose[:2]) - np.array(obstacle_pos))
                # print('check = ', d_to_obstacle)
                if d_to_obstacle < min_distances_to_obstacles[i]:
                    min_distances_to_obstacles[i] = d_to_obstacle
                    min_poses_to_obstacles[i] = pose
        # for i, obstacle_pos in enumerate(obstacles_pos):
        #     print(f"Minimum distance to obstacle {i+1} across {drone}: {min_distances_to_obstacles[i]} at pose {min_poses_to_obstacles[i]}")

if __name__ == "__main__":
    main()                 