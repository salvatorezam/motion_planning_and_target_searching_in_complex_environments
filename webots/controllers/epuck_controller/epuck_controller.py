from controller import Supervisor
import numpy as np
from numpy import inf
import math
import scipy.cluster.hierarchy as hcluster


# ----------------- GLOBAL PARAMETERS -----------------

# the robot of the world is not a direct Webots PROTO, rather, it is a ROBOT node resembling an epuck;
# this choice is justified by the fact that this is the configuration than enables a more flexible 
# customization of the robot in terms of the embeddable auxiliary devices as the LIDAR and camera. It 
# follows that we will need to explicitly export some parameters related to the physical construction 
# of the epuck directly from the cyberbotics docs: https://cyberbotics.com/doc/guide/epuck
#
ROBOT_WHEEL_RADIUS = 0.0205
ROBOT_AXLE_LENGTH = 0.052 
ROBOT_MAX_VELOCITY = 6.28

# this variable determines the operational mode of the robot in each of the simulation iteration:
#   - DEFAULT: the robot turns to the goal direction and proceeds forward
#   - AVOID_OBSTACLE: an obstacle is detected between the robot and the goal -> avoid the obstacle
#   - FLANK_OBSTACLE: the robot flanks the obstacle that it just surpassed
#   - SLAM: the detection of the reference_point_obj enables the SLAM mode
#
ACTION_TYPE = 'DEFAULT'

# initialization of the variables for standard deviation and uncertainty 
# of the measures to be fed to the Extended Kalman Filter algorithm
#

# standard deviations associated with the linear and angular velocity
sigma_n_v = 0.01
sigma_n_omega = np.pi/60

# covariance matrix
Sigma_n = np.array([[sigma_n_v**2, 0.0],
                    [0.0, sigma_n_omega**2]])

# other utility parameters
#
OBSTACLE_PROXIMITY_THRESHOLD = 0.1
SLAM_TO_DEFAULT = False
has_obstacle_been_avoided = False
mode_dir = -1
rubber_duck_position = [-2.9, 0.747, 0.0]



# ----------------- UTILITY FUNCTIONS -----------------

# normalizes the input angle in the [0, 2*pi] interval
def normalize_angle(angle):
    norm_angle = angle
    norm_angle += 2 * np.pi if norm_angle < 0 else 0
    return norm_angle

# returns the index of the group element with minimum distance from value (its nearest neighbor)
def nn(group, value):
    group = np.asarray(group)
    return (np.abs(group - value)).argmin()

# returns the rotation matrix associated to the angle
def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

# computes the bearing measure on the basis of the position 
# of the object detected by the camera, its parameters are:
#   - ref_point_horizontal_pos: horizontal location of the object in the captured image
#   - image_width: width of the captured image 
#
def bear_meas_with_img(ref_point_horizontal_pos):
    d = (0.5 * camera_width) / np.tan(0.5 * camera_field_of_view)
    return np.arctan2(0.5 * camera_width - ref_point_horizontal_pos, d)

# Hough transform for mapping LIDAR data
def hough_transform(distances):

    # computing the angle values of the individual LIDAR scannings
    angles_max_res = np.array([i * (lidar_field_of_view / horizontal_resolution) for i in range(horizontal_resolution)])
    angles_cl_res = np.array([i * (lidar_field_of_view / 90) for i in range(90)])

    # finding the greatest registererd distance that is not associated with a 
    # scanning towards a zone free of obstacles (infinite detected distance)
    distances[distances == inf] = 0
    max_distance = np.amax(distances)

    # the distance intervals are partitioned according to the maximum distance
    # so that equally-spaced samples are retrieved in [-max_distance, +max_distance]
    len_part_dist, half_lpd = 100, 50
    part_dist = np.zeros(len_part_dist)
    for i in range(len_part_dist):
        if i <= half_lpd:
            part_dist[i] = (half_lpd - i) * (-1 * max_distance / half_lpd)
        else:
            part_dist[i] = (i - half_lpd) * (max_distance / half_lpd)

    # quantifying, for each sampled distance, the number of radii with maximum resolution close to it
    near_mat = np.zeros((len_part_dist, len(angles_cl_res)))
    for i in range(len(distances)):
        for j in range(len(angles_cl_res)):
            nearest = nn(part_dist, distances[i] * np.cos(angles_cl_res[j] - angles_max_res[i]))
            near_mat[nearest, j] += 1

    # the indexes associated with the greatest proximities are parsed
    nearest_rows_indices, nearest_col_indices = np.where(near_mat >= 70)

    # filling the radii list by extracting those related to the closest indexes
    rays = np.zeros((2,len(nearest_rows_indices)))
    for i in range(len(nearest_rows_indices)):
        rays[0][i] = part_dist[nearest_rows_indices[i]]
        rays[1][i] = angles_cl_res[nearest_col_indices[i]]

    return rays.T



# ----------------- EKF ALGORITHM -----------------

# our implementation of the EKF algorithm is based on the following definitions:
#   - https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf
#   - https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48

#
# this function performs the prediction stage of the Kalman filter, with the objective of
# estimating the actual new position reached by the robot;
# 
# its parameters are, respectively:
#   - updated_position: array in the form [x, y, theta], associated with the initial position 
#     values that the filter needs to update
#   - Sigma_p: estimation of the uncertainty of the position and angle
#   - u: control signals
#   - Sigma_n: uncertainty on the control signals
#   - iteration_timestep: duration of one iteration of the simulation
#
def predict_ext_kalman_filter(updated_position, Sigma_p, u, Sigma_n, iteration_timestep):
    
    # evaluating the new position and orientation according to the EKF algorithm
    x = updated_position[0] + u[0] * np.cos(updated_position[2]) * iteration_timestep
    y = updated_position[1] + u[0] * np.sin(updated_position[2]) * iteration_timestep
    theta = updated_position[2] + u[1] * iteration_timestep
    updated_position = np.array([x, y, theta])

    # evaluating the uncertainty on the basis of the average on the previously detected distances
    phi = np.asarray([[1, 0, -v * np.sin(updated_position[2]) * iteration_timestep], 
                      [0, 1, v * np.cos(updated_position[2]) * iteration_timestep], 
                      [0,0,1]])
    G = np.asarray([[np.cos(updated_position[2]) * iteration_timestep, 0], 
                    [np.sin(updated_position[2]) * iteration_timestep, 0], 
                    [0,iteration_timestep]])

    Sigma_p = phi @ Sigma_p @ phi.T + G @ Sigma_n @ G.T

    return updated_position, Sigma_p

#
# update stage of the EKF algorithm; the function returns the updated position together with its uncertainty (variance)
# according to the measures passed as input (along with their uncertainty as well) and and to the realtive position
# of the robot with respect to the objects detected by its camera
#
# particularly, it takes as input:
#   - updated_position: the last updated position (and orientation) value
#   - Sigma_p: estimation of the uncertainty of the position
#   - z: parsed measures
#   - Sigma_m: uncertainty on measures
#   - rec_obj_global_pos: (global) position of the detected objects
#   - iteration_timestep: duration of one iteration of the simulation
#
def relative_pos_update_EKF(updated_position, Sigma_p, z, Sigma_m, rec_obj_global_pos):

    updated_position = np.asarray(updated_position)
    Sigma_p = np.asarray(Sigma_p)
    z = np.asarray(z)
    Sigma_m = np.asarray(Sigma_m)
    rec_obj_global_pos = np.asarray(rec_obj_global_pos)
    
    # the current position parameters are initialized
    pos = np.asarray(updated_position[0:2])
    theta = np.asarray(updated_position[2])

    transp_theta_rot_mat = rotation_matrix(theta).T

    # EKF update
    z_hat = transp_theta_rot_mat @ (np.array(rec_obj_global_pos)[0:2]-pos)
    delta_z = z - z_hat
    top = - transp_theta_rot_mat
    bottom = (- transp_theta_rot_mat @ np.array([[0,-1],[1,0]]) @ (np.array(rec_obj_global_pos)[0:2]-pos)).reshape((2,1))
    H = np.hstack((top,bottom))

    S = H @ Sigma_p @ H.T + Sigma_m
    K = Sigma_p @ H.T @ np.linalg.inv(S)
    updated_position = updated_position + (K @ delta_z)
    Sigma_p = Sigma_p - (K @ H @ Sigma_p)

    return updated_position, Sigma_p



# ----------------- ROBOT (SUP) AND DEVICES INITIALIZATION -----------------

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

ROBOT_WHEEL_RADIUS = 0.0205
ROBOT_AXLE_LENGTH = 0.053 
ROBOT_MAX_VELOCITY = 6.28

# - motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

# - positioning
left_ps = robot.getDevice('left_wheel_sensor')
right_ps = robot.getDevice('right_wheel_sensor')

left_ps.enable(timestep)
right_ps.enable(timestep)

# - lidar
lidar = robot.getDevice('lidar')
lidar.enable(1)

# parsing the lidar fov and resolution, that are set 
# respectively to 3.14 and 512 in the Webots world
lidar_field_of_view = lidar.getFov()
horizontal_resolution = lidar.getHorizontalResolution()

# the deafult behavior of the lidar object is that of returning the depth values (from left to right)
# on an array, along with those from the upper to lower layers; the 'point cloud' mode allows visualizing 
# in the simulation the point cloud detected by the device
#
# https://cyberbotics.com/doc/reference/lidar#wb_lidar_enable_point_cloud
#
lidar.enablePointCloud()

# - camera
camera = robot.getDevice('camera')
camera.enable(1)

# storing the fov and image width of the camera, that are 
# set respectively to 0.84 and 320 in the Webots world
camera_width = camera.getWidth()
camera_field_of_view = camera.getFov()

# the 'camera' node of the Webots simulation has a 'recognition' child node that allows to take advantage 
# of the object recognition feature; in addition, we will also need to explicitlt enable the segmentation
# (with the same sampling rate of the recognition)
#
# https://cyberbotics.com/doc/reference/camera?tab-language=python#wb_camera_recognition_enable_segmentation
#
camera.recognitionEnable(1)
camera.enableRecognitionSegmentation()



# ----------------- EPUCK POSITION VARIABLES INITIALIZATION -----------------

epuck = robot.getFromDef("e-puck")

init_position, init_orientation = epuck.getPosition(), epuck.getOrientation()

x_s, y_s = init_position[:2]

orientation = np.arctan2(init_orientation[3],init_orientation[0])
orientation = normalize_angle(orientation)

prev_theta_start = np.arctan2(init_orientation[3],init_orientation[0])
prev_theta_start = normalize_angle(prev_theta_start)

x_s_est, y_s_est = init_position[:2]

iteration_timestep = timestep / 1000

pos_sens_values = [0,0]
distance_values = [0,0]
prev_pos_sens_values = [0,0]
differs = [0,0]



# ----------------- MAIN LOOP -----------------
while robot.step(timestep) != -1:

    # parsing the information on starting position, orientation and velocity
    #
    init_position, init_orientation, velocity = epuck.getPosition(), epuck.getOrientation(), epuck.getVelocity()
    
    # explicitly evaluating the starting orientation angle, normalizing 
    # it into the [0, 2*pi] interval in the case that it is not already
    #
    theta_start = normalize_angle(np.arctan2(init_orientation[3], init_orientation[0]))

    # computing the distance travelled by the robot through dead reckoning,
    # starting from the registered position sensors of each wheel
    #
    pos_sens_values = [left_ps.getValue(), right_ps.getValue()]

    for i in range(len(pos_sens_values)):

        delta_ps_vals = pos_sens_values[i] - prev_pos_sens_values[i]

        # the PositionSensor of the simulation has its 'resolution' field set to -1 as default,
        # encoding an 'infinite' resolution (the maximum possible one); as this information may be 
        # exploited when determining the travelled distance, there are some cases (external factors,
        # robot's inertia, etc.) where this could be unnecessary, if not wrong; this is the reason why 
        # we will fix at 0.001 the threshold below which the travelled distance can be considered zero
        if delta_ps_vals < 0.001:
            delta_ps_vals = 0
            pos_sens_values[i] = prev_pos_sens_values[i]
        distance_values[i] = delta_ps_vals * ROBOT_WHEEL_RADIUS

    v = (distance_values[0] + distance_values[1])/ (2.0)
    w = (distance_values[1] - distance_values[0])/ (ROBOT_AXLE_LENGTH)

    # the reached position is dead-reckoned as the product of the 
    # robot's velocity and the time interval (of an iteration)
    x_s_est += velocity[0] * iteration_timestep
    y_s_est += velocity[1] * iteration_timestep

    # similarly, we compute the rotation angle and initialize its associated rotation matrix
    orientation += velocity[5] * iteration_timestep
    orientation = normalize_angle(orientation)

    rot = rotation_matrix(orientation)
    
    # the current position gets stored
    prev_pos_sens_values = pos_sens_values.copy()

    # the robot moves towards the goal by first rotating on itself until the forward direction overlays
    # the line that connects it to the target and, subsequently, by proceeding linearly along this direction
    #
    # https://levelup.gitconnected.com/webots-series-move-your-robot-to-specific-coordinates-ecf50cb4244b
    #

    # angle between the robot and the target
    theta = np.arctan2(rubber_duck_position[1] - init_position[1], rubber_duck_position[0] - init_position[0])  % (2*np.pi)

    # angle between the target and the frontal direction, the one which the rotation will occur on
    delta_theta = theta - theta_start

    # reading lidar values
    lidar_values = np.array(lidar.getRangeImage())

    # extracting the individual scannings of the lidar
    lidar_rays = hough_transform(np.copy(lidar_values))

    # in order to clean and preprocess the lidar data, we perform a clustering on its values;
    # fclusterdata executes the clustering on the specified metric ('euclidean' by default)
    # in a hierarchical way through the single linkage approach. Then, it forms flat clusters 
    # according to the inconsistency on the threshold
    #
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html
    #
    cutoff_threshold = 0.05
    clusters = hcluster.fclusterdata(lidar_rays, cutoff_threshold, criterion="distance")

    # combining and restructuring the computed clustering
    stacked_lc = np.hstack((lidar_rays, clusters[:,None]))
    stacked_lc = stacked_lc[stacked_lc[:, -1].argsort()]
    lc = np.split(stacked_lc[:,:-1], np.unique(stacked_lc[:, -1], return_index=True)[1][1:])

    # for each of the clusters, we find the relative 
    # centroid trhough the average on the elements of lc
    centroids = []
    for i in range(len(lc)):
        m = np.mean(lc[i], axis=0)
        centroids.append(m)

    b = np.asarray(centroids)

    # obstacle detection through the centroids of the reference lidar rays
    for i in range(len(centroids)):

        # parsing the central values of the lidar scanning and marking 
        # those that go to infinity (absent or distant obstacle)
        central_values = lidar_values[horizontal_resolution//2-30:horizontal_resolution//2+30]
        central_values[central_values == inf] = 1

        # if the lidar detects that the robot is getting (frontally) close to a wall,
        # it stops to avoid the collision and triggers the AVOID_OBSTACLE mode
        if np.mean(central_values) <= 0.5 and np.abs(centroids[i][0]) != 0.0 and np.abs(centroids[i][0]) <= OBSTACLE_PROXIMITY_THRESHOLD:

            if np.abs(delta_theta) < 0.05:
                ACTION_TYPE = 'AVOID_OBSTACLE'
                near_obstacle = np.abs(centroids[i])

                right_motor.setVelocity(0)
                left_motor.setVelocity(0)

                break

    # switch case to determine the operational mode of the robot (DEFAULT, AVOID_OBSTACLE_FLANK_OBSTACLE, SLAM)
    #

    # the DEFAULT action is that related to making the robot orient itself 
    # in the direction the target and proceed in a straight line towards it
    if ACTION_TYPE == 'DEFAULT':

        # if the robot is not oriented towards the goal, it turns left or right according to its relative angle
        if np.abs(delta_theta) > 0.05:
            print(f'Operational mode {ACTION_TYPE} - rotating towards goal.')

            # turn left
            if delta_theta > 0:
                left_motor.setVelocity(-1.2)
                right_motor.setVelocity(1.2)
                
            # turn right
            if delta_theta < 0: 
                left_motor.setVelocity(1.2)
                right_motor.setVelocity(-1.2)

        # the robot is oriented towards the goal and it can now go forward
        else:
            print(f'Operational mode {ACTION_TYPE} - proceeding towards goal.')
            right_motor.setVelocity(ROBOT_MAX_VELOCITY)
            left_motor.setVelocity(ROBOT_MAX_VELOCITY)

    # in AVOID_OBSTACLE, the lidar has detected an obstacle in front of the robot, which will 
    # now need to change direction in order to avoid it and keep going towards the target
    elif ACTION_TYPE == 'AVOID_OBSTACLE':
        print(f'Operational mode {ACTION_TYPE} - avoiding obstacle.')

        # evaluating the (frontal) angle between the starting one and 
        # the radius (centroid) associated with the detected obstacle
        delta_par = near_obstacle[1] - theta_start % np.pi
        if np.abs(delta_par) >= 0.03:

            # the external values of the lidar scanning are extracted and, in the braitenberg fashion,
            # the rotation direction is evaluated as the one that moves the robot in the freeer area
            left_corner_scan, right_corner_scan = lidar_values[0:30], lidar_values[horizontal_resolution-30:horizontal_resolution]

            left_corner_scan[left_corner_scan == inf] = 100
            right_corner_scan[right_corner_scan == inf] = 100

            left_corner_scan, right_corner_scan = np.mean(left_corner_scan), np.mean(right_corner_scan)

            if left_corner_scan < right_corner_scan and mode_dir == -1:
                mode_dir = 1

            # turn right
            if mode_dir == 1:
                
                left_motor.setVelocity(1)
                right_motor.setVelocity(-1)
            
            # turn left
            else:
                mode_dir = 0
                left_motor.setVelocity(-1)
                right_motor.setVelocity(1)
                
        # if delt_par is small (zero), it means that the robot's direction 
        # is parallel to the obstacle, and it can now proceed to flank it
        else:
            right_motor.setVelocity(0)
            left_motor.setVelocity(0)

            ACTION_TYPE = 'FLANK_OBSTACLE'
            mode_dir = -1

    # the FLANK_OBSTACLE mode refers to the situation where the robot has moved away from 
    # the obstacle and it coasts it to surpass it and get back to the target's direction
    elif ACTION_TYPE == 'FLANK_OBSTACLE':
        print(f'Operational mode {ACTION_TYPE} - flanking obstacle.')

        right_motor.setVelocity(6)
        left_motor.setVelocity(6)

        centr_np = np.asarray(centroids)
        centr_np[centr_np == 0] = 100

        # if the robot is sufficiently far from the obstacle and is still in the 
        # FLANK_OBSTACLE mode, it gets ready to get back to the DEFAULT behavior
        if (np.abs(centr_np[np.argmin(np.abs(centr_np[:,0]))][0]) >= near_obstacle[0] + 0.05) and has_obstacle_been_avoided == False:
            has_obstacle_been_avoided = True
            back_on_track_counter = 0

        elif has_obstacle_been_avoided == True:
            back_on_track_counter += 1
            if back_on_track_counter == 40:
                has_obstacle_been_avoided = False
                ACTION_TYPE = 'DEFAULT'
                
    # the SLAM mode gets into play when the robot cannot directly explot the lidar information on the walls of the 
    # environment it is located in; instead, it uses the information it gathers from the landmarks to locate itself
    # and ultimately move towards the target
    elif ACTION_TYPE == 'SLAM':
        print(f'Operational mode {ACTION_TYPE} - SLAM.')
        
        if is_slam_mode_triggered == True:
            slam_counter += 1
            if slam_counter == 10:
                is_slam_mode_triggered = False

        else:

            print('SLAM computation.')

            # parsing the information on the linear and rotational velocities
            r_velocity = np.copy(epuck.getVelocity())
            u = np.array([np.linalg.norm(r_velocity[:2]), r_velocity[5]])

            # the new (estimated) position is computed, together with its uncertainty,
            # all through the predict stage of the EKF algorithm
            updated_position, Sigma_p = predict_ext_kalman_filter(updated_position, Sigma_p, u, Sigma_n, iteration_timestep)

            # parsing the objects that are detected by the robot's camera
            recognized_objects = camera.getRecognitionObjects()
            num_recognized_objects = camera.getRecognitionNumberOfObjects()

            relative_pos_meas = np.zeros((num_recognized_objects, 2)) 

            far_dist, far_pos, far_bear = -1, -1, -1

            # updating the actual position through the one relative to the detected landmarks
            #
            for i in range(0, num_recognized_objects):

                # the single object and its position are extracted
                reference_point_obj = robot.getFromId(recognized_objects[i].get_id())
                rec_obj_global_pos = reference_point_obj.getPosition()

                # evaluating bear_meas on the basis of the position of the object on the image
                bear_meas = bear_meas_with_img(recognized_objects[i].get_position_on_image()[0])

                # finding in which sector of the lidar scan the object is located 
                # and extracting the relative distance to the rays that hit it
                lidar_bear = np.round(256 - (bear_meas / np.pi) * 512)
                lidar_dist = np.array(lidar.getRangeImage())[int(lidar_bear)]

                # if either the object is distant, or there has been a false
                # detection, we do not involve it in the SLAM phase
                if lidar_dist == inf:
                    pass
                else:
  
                    # retrieving the coordinates of the object based on the distance 
                    # detected by the lidar and the angle of the scan ray
                    ref_point_position = np.array([lidar_dist * np.cos(bear_meas), 
                                                   lidar_dist * np.sin(bear_meas)])

                    # explicitly evaluating the position relative to the object
                    translated_pos = rot @ ref_point_position + updated_position[:2]
                    translated_pos = np.append(translated_pos,0.05)

                    rec_obj_global_pos_2 = [translated_pos[0], 
                                            translated_pos[1], 
                                            0.05]

                    # finding the parameters of the farthest detected object
                    if lidar_dist > far_dist:
                        far_dist = lidar_dist
                        far_pos = recognized_objects[i].get_position()
                        far_bear = bear_meas

                    # getPose(node) returns an array of 16 elements to be interpreted as a 4x4 transformation 
                    # matrix representing a relative transformation on the node it is called on
                    # 
                    # https://cyberbotics.com/doc/reference/supervisor#wb_supervisor_node_get_pose
                    # 
                    rel_lm_trans = reference_point_obj.getPose(epuck)

                    # standard deviation
                    std_m = 0.05
                    Sigma_m = [[std_m**2, 0], 
                               [0, std_m**2]]
          
                    # kalman estimate on the gaussian of the robot's position uncertainty
                    relative_pos_meas[i] = [ref_point_position[0] + np.random.normal(0,std_m), 
                                            ref_point_position[1] + np.random.normal(0,std_m)]

                    # update stage of the EKF algorithm
                    updated_position, Sigma_p = relative_pos_update_EKF(updated_position, Sigma_p, relative_pos_meas[i], Sigma_m, rec_obj_global_pos_2)

            # if the farthest object is in close proximity to the 
            # robot, then it slows doen in order to avoid collision
            if far_dist <= 0.05:
                left_motor.setVelocity(1.0)
                right_motor.setVelocity(1.0)

            # if the robot is free to proceed forward, then it does it 
            # proportionally to the distances of the reference points
            else:

                # 5.0 gain
                g = 5.0 * far_dist

                # restricting the velocity within its maximum allowed value
                v = g if g <= ROBOT_MAX_VELOCITY else ROBOT_MAX_VELOCITY

                # angle between the landmark and the goal
                delta_target_angle = ((math.atan((far_pos[1]) / (far_pos[0])) + math.pi) % (2 * math.pi)) - math.pi

                # 30.0 gain
                w = 30.0 * delta_target_angle

                # finding the horizontal steering velocities with the epuck's ROBOT_AXLE_LENGTH, 
                # checking that they do not go over the maximum allowed one 
                new_left_vel, new_right_vel = v - (w * ROBOT_AXLE_LENGTH * 0.5), v + (w * ROBOT_AXLE_LENGTH * 0.5)

                vl = new_left_vel if new_left_vel <= ROBOT_MAX_VELOCITY else ROBOT_MAX_VELOCITY
                vr = new_right_vel if new_right_vel <= ROBOT_MAX_VELOCITY else ROBOT_MAX_VELOCITY

                # setting the motors' velocities
                left_motor.setVelocity(vl)
                right_motor.setVelocity(vr)

            # in the case that only one object is detected, it means that the robot is either entering
            # or exiting a SLAM zone, therefore the DEFAULT behavior needs to be triggered
            if num_recognized_objects == 1:
                ACTION_TYPE = 'DEFAULT'
                SLAM_TO_DEFAULT = True
                
    # the robot enters the SLAM mode if it is not already in it (ACTION_TYPE != 'SLAM') or it just 
    # exited it (not SLAM_TO_DEFAULT) and the camera has detected a minimum number (3) of objects
    if ACTION_TYPE != 'SLAM' and not SLAM_TO_DEFAULT and len(camera.getRecognitionObjects()) >= 3:

        # we make sure that the robot enters the SLAM zone
        right_motor.setVelocity(ROBOT_MAX_VELOCITY)
        left_motor.setVelocity(ROBOT_MAX_VELOCITY)

        print('SLAM phase initialization.')

        # the parameters are reinitialized
        init_position, init_orientation = epuck.getPosition(), epuck.getOrientation()
        x_s, y_s = init_position[:2]
        theta_start = normalize_angle(np.arctan2(init_orientation[3],init_orientation[0]))
        updated_position = [x_s, y_s, theta_start]

        Sigma_p = np.array([[0.01, 0.0, 0.0],
                            [0.0, 0.01, 0.0],
                            [0.0, 0.0, np.pi/90]])
                            
        is_slam_mode_triggered = True
        slam_counter = 0

        # performing the operational mode change
        ACTION_TYPE = 'SLAM'

    pass
