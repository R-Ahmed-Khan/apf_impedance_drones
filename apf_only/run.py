from apf_only_path import*
from apf import*
from pprint import pprint

def get_drone_poses():
    '''
    num_drones = 4 (default)
    start_pos = initial point of APF trajectory
    goal_pos = goal point of  APF trajectory
    obstacles_pos = position of obstacles, add as much obstacles required in the given format e.g. [[2,2.9],[2.9,2],[obs_new_x,obs_new_y]]
    r_apf_list = radius list of APF repulsion field around obstacles
    r_imp_list = radius list of repulsive force applied to any drone in vicinty around obstacles
    d_sep = seperation b/w drones 
    step = step size of trajectory
    trail_drones = [1,2] ==> plots trails of drone 1 and 2 only
    rotation = 5*np.pi/4 ==> drone 1 is most forward
    '''
    instance = APF_ONLY()      
    
    ## case #4 -- obstacles only -- more cluttered 
    obstacles_pos = [[-2.5,2],[-1,2],[-1,0.6],[0,-0.4],[1,0.6]]  
    drone_poses_dict = instance.simulate(num_drones=4, start_pos=(-3.0,3.0), goal_pose=(2,-1), \
                                            obstacles_pos=obstacles_pos, \
                                               r_apf_list=[0.3,0.3,0.3,0.3,0.3], \
                                                d_sep=0.6,step=0.01, plot=True,trail_drones=[1,2,4], rotation = 5*np.pi/4)
    
    # pprint(drone_poses_dict)
    # plot_drone_poses(drone_poses_dict)
    
    return drone_poses_dict, obstacles_pos

def main():

    drone_poses_dict, obstacles_pos= get_drone_poses()

    # print('shape = ', drone_poses_dict.shape[0])
    # Save drone poses dictionary to a file using pickle
    preprocessed_drone_poses = {}
    for drone, poses in drone_poses_dict.items():
        preprocessed_poses = [] 
        for pose in poses:
            preprocessed_poses.append(' '.join(map(str, pose)))
        preprocessed_drone_poses[drone] = '\n'.join(preprocessed_poses)

    # Save preprocessed drone poses dictionary to a file
    with open('apf_only/drone_poses.txt', 'w') as f:
        for drone, poses in preprocessed_drone_poses.items():
            f.write(f"{drone}:\n{poses}\n\n")
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