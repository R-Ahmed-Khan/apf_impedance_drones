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
    trail_drones = [1,2] ==> plots trails of drone 1 and 2 only
    rotation = 5*np.pi/4 ==> drone 1 is most forward
    '''
    instance = APF_IMP()
    # case #1 -- gates only 
    # obstacles_pos = [[0.03, 1.03],[-1.03, -0.03]]  
    obstacles_pos = [[-0.12, 0.65],[-0.88, -0.65]]  
    drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,1.5), goal_pose=(2.0,-1.5), obstacles_pos=obstacles_pos, \
                                               r_apf_list=[0.32,0.32], r_imp_list=[0.26,0.26], \
                                                d_sep=0.60, step=0.02, plot=True, trail_drones=[1,2],rotation=0) 

    # case #2 -- obstacles only 
    # obstacles_pos = [[-1.0,1.0],[1.0, 0.0]]  
    # obstacles_pos = [[-2.0,0.6],[0.5, -0.4]]  
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,1.5), goal_pose=(2.0, -1.5), obstacles_pos=obstacles_pos, \
    #                                           r_apf_list=[0.32,0.32], r_imp_list=[0.22,0.22], \
    #                                            d_sep=0.60, step=0.02, plot=True, trail_drones=[3,4],rotation=0)       

    ## case #3 -- obstacles and gates  
    # obstacles_pos = [[0.53, 1.03],[-0.53, -0.03], [-1.0,1.0],[1.0, 0.0]]  
    # obstacles_pos = [[-0.12, 0.65],[-0.88, -0.65],[-2.0,0.6],[0.5, -0.4]]  

    # experimental
    # obstacles_pos = [[-1.258,0.097],[1.221, -0.708],[-0.020,0.522],[-0.424, -0.712]]  
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,1.5), goal_pose=(2.294, -1.531), obstacles_pos=obstacles_pos, \
    #                                             r_apf_list=[0.32,0.32,0.32,0.32], r_imp_list=[0.26,0.26,0.22,0.22], \
    #                                             d_sep=0.60, step=0.02, plot=True, trail_drones=[3,4],rotation=0)       
    
    ## case #4 -- obstacles only -- more cluttered 
    # obstacles_pos = [[-2.5,2],[-1,2],[-1,0.6],[0,-0.4],[1,0.6]]  
    # r_apf_list=[0.45,0.45,0.45,0.45,0.45], r_imp_list=[0.28,0.28,0.28,0.28,0.28] , start_pos=(-3.0,3.0), goal_pose=(2.0,-1.0)

    # experimental
    # obstacles_pos = [[-1.075, 0.327],[0.458, -0.973],[0.493,0.436],[-0.804, -0.679]] # start_pos=(-3.0,1.5), goal_pose=(2.294, -1.531),
    # drone_poses_dict, path = instance.simulate(num_drones=4, start_pos=(-3.0,1.5), goal_pose=(2.294, -1.531), obstacles_pos=obstacles_pos, \
    #                                            r_apf_list=[0.35,0.32,0.32,0.32], r_imp_list=[0.26,0.26,0.22,0.22], \
    #                                             d_sep=0.60, step=0.01, plot=True, trail_drones=[2,4], rotation=0)
    
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