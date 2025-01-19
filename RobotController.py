from cmath import atan, tan
import sys
import time
import random
import math
import numpy as np
import cv2 # type: ignore
sys.path.append('C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python\src') 
from coppeliasim_zmqremoteapi_client import RemoteAPIClient # type: ignore

# This code connects to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')
print('Connected to CoppeliaSim')

# This code starts the simulation
sim.startSimulation()
print("Simulation started.")

# Get Object Handle is basically the handle of the robot we want to control
handle = sim.getObject('/NiryoOne') # Robot name is NiryoOne
print(f"Robot handle: {handle}")

# Getting the base
Robot_base = sim.getObject('/NiryoOne/visible')
print(f"Base handle: {Robot_base}")

# Getting the Gripper
Robot_gripper = sim.getObject('/NiryoOne/NiryoLGripper')
print(f"Gripper handle: {Robot_gripper}")

# Getting Vision Sensor
Vision_sensor = sim.getObject('/NiryoOne/Vision_sensor')
print(f"Vision Sensor handle: {Vision_sensor}")

# Getting Proximity Sensor
Proximity_sensor = sim.getObject('/NiryoOne/NiryoLGripper/Proximity_sensor')
print(f"Proximity Sensor handle: {Proximity_sensor}")

# Getting the joints
Robot_joints = [ 
    sim.getObject('/NiryoOne/Joint'),
    sim.getObject('/NiryoOne/Link/Joint'),
    sim.getObject('/NiryoOne/Link/Joint/Link/Joint'),
    sim.getObject('/NiryoOne/Link/Joint/Link/Joint/Link/Joint'),
    sim.getObject('/NiryoOne/Link/Joint/Link/Joint/Link/Joint/Link/Joint'),
    sim.getObject('/NiryoOne/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint'),
]

# Configuration function for the robot Arm
def move_to_config(joints, maxVel, maxAccel, maxJerk, target_config):
    params = {
        'joints': joints,
        'targetPos': target_config,
        'maxVel': maxVel,
        'maxAccel': maxAccel,
        'maxJerk': maxJerk
    }
    sim.moveToConfig(params)

# Converts Degrees to Radian
def deg_to_radian(x):
    return x * 3.14159 / 180

# getting ditance between two points
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[0])**2)

# This is for randomly placing bricks in the robot's workspace and dummies (needed for IK)!
def place_bricks(base, arm_position, max_radius, placed_bricks):

    # Getting Base position
    base_position = sim.getObjectPosition(base, -1)

    # Getting size of brick (length x width x height)
    brick_size = [0.04, 0.04, 0.04]

    min_distance = 0.35
    max_attempts = 100

    # Two different locations for the regions for the bricks to be spawned in
    region1 = {'x': [max_radius * 0.6,max_radius * 0.9], 'y': [-0.1, 0.1]}
    region2 = {'x': [max_radius * 0.4,max_radius * 0.8], 'y': [-0.1, 0.1]}

    selected_region = random.choice([region1, region2])

    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        # Getting location of brick
        angle = random.uniform(0, 2 * math.pi)  # Random angle for radial distribution
        radius = random.uniform(max_radius * 0.5, max_radius * 0.8)

        x_offset = radius * math.cos(angle)
        y_offset = radius * math.sin(angle)

        x_offset += random.uniform(selected_region['x'][0], selected_region['x'][1])
        y_offset += random.uniform(selected_region['y'][0], selected_region['y'][1])
        z_offset = brick_size[2] / 2 + 0.2

        #brick_position = [base_position[0] + x_offset, base_position[1] + y_offset, base_position[2] + z_offset]
        brick_position = [0.33147, 0.4317, 0.04]

        valid_position = True
        for existing_brick in placed_bricks:
            existing_pos = sim.getObjectPosition(existing_brick, -1)
            if calculate_distance(brick_position, existing_pos) < min_distance:
                valid_position = False
                break

        if valid_position:

            #brick_orientation = [roll, pitch, yaw]
            brick_orientation = [0,0,0]

            # Create Brick
            brick = sim.createPrimitiveShape(sim.primitiveshape_cuboid, brick_size, 0)
            sim.setObjectPosition(brick, -1, brick_position)
            sim.setObjectOrientation(brick, -1, brick_orientation)

            # Colours for Brick
            colour = [random.random(), random.random(), random.random()]
            sim.setShapeColor(brick, None, sim.colorcomponent_ambient, colour)

            # Create dummy
            dummy = sim.createDummy(0.02, colour)
            sim.setObjectParent(dummy, brick, True)
            sim.setObjectPosition(dummy, brick, [0, 0, brick_size[2]/4])

            # Brick Settings
            floor = sim.getObject('/Floor')

            sim.setObjectInt32Param(brick, sim.shapeintparam_respondable, 1)
            sim.setObjectInt32Param(floor, sim.shapeintparam_respondable, 1)

            # Making brick dynamic
            sim.setObjectInt32Param(brick, sim.shapeintparam_static, 0)
            sim.setObjectFloatParam(brick, sim.shapefloatparam_mass, 0.2)

            # Making brick and Floor collidable :)
            sim.setObjectSpecialProperty(brick, sim.objectspecialproperty_collidable)
            sim.setObjectSpecialProperty(floor, sim.objectspecialproperty_collidable)

            # Slowed down the bricks
            sim.setArrayParam(sim.arrayparam_gravity, [0, 0, -2.5])

            return brick, dummy
    return None

# Getting the image from vision_data
def Image_retriver(vision_data, width, height):

    image_data = vision_data[0]
    unpacked_data = sim.unpackUInt8Table(image_data)

    if isinstance(unpacked_data, tuple):
        unpacked_data = unpacked_data[0]

    image = np.array(unpacked_data, dtype=np.uint8).reshape((height, width, 3))
    return image

# Detecting blocks from the image
def detect_blocks(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # If the contour has 4 vertices, it's likely a rectangular block
        if len(approx) == 4:
            blocks.append(approx)
    
    return blocks

# Getting the block position
def getBlock_Position(block):
    M = cv2.moments(block)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    return None
    
# Getting proximity distance from the proximity sensor
def getProximityDistance(proximity_sensor):

    # Getting the distance
    result = sim.readProximitySensor(proximity_sensor)
    detection, detected_point = result[:2]
    if detection and detected_point:
        return detected_point
    else:
        return None


def close_gripper():
    sim.setJointTargetPosition(Robot_gripper, 1)

def open_gripper():
    left_joint = sim.getObject('/NiryoOne/NiryoLGripper/leftJoint1')
    right_joint = sim.getObject('/NiryoOne/NiryoLGripper/rightJoint1')

    sim.setJointTargetPosition(left_joint, -1.39)
    sim.setJointTargetPosition(right_joint, +1.39)
    sim.wait(2.0)

def close_gripper():
    left_joint = sim.getObject('/NiryoOne/NiryoLGripper/leftJoint1')
    right_joint = sim.getObject('/NiryoOne/NiryoLGripper/rightJoint1')

    sim.setJointTargetPosition(left_joint, 0)
    sim.setJointTargetPosition(right_joint, 0)
    sim.wait(2.0)

def smooth_move(robot_tip, target_pos, target_ori, waypoints=[], duration=7.0):
    waypoints.append(target_pos)
    
    # Move smoothly through each waypoint
    for waypoint in waypoints:
        sim.setObjectPosition(robot_tip, -1, waypoint)
        sim.setObjectOrientation(robot_tip, -1, target_ori)
        sim.wait(duration)

# Moving the robot
def move_to_position(robot_tip, target_position, target_orientation):

    # Open the gripper
    open_gripper()

    # check for the orientation here!
    sim.setObjectOrientation(robot_tip, -1, target_orientation)
    sim.setObjectPosition(robot_tip, -1, target_position)
    sim.wait(5.0)

    # Close the gripper
    close_gripper()


# Moving the robot
def move_to_plane(robot_tip, target_position, target_orientation):

    waypoints = [
        [target_position[0], target_position[1], target_position[2] + 0.01],  # Slightly above the target
        target_position
    ]
    smooth_move(robot_tip, target_position, target_orientation, waypoints, duration=1.0)
    sim.wait(2)

    # Open the gripper
    open_gripper()

# Moving the robot
def move_to_default(robot_tip, target_position, target_orientation):

    waypoints = [
        [target_position[0], target_position[1], target_position[2] + 0.05],
        target_position
    ]
    smooth_move(robot_tip, target_position, target_orientation, waypoints, duration = 1.0)
    

# This checks whether the gripper is close or not
def is_gripper_close(gripper_position, block_position, threshold=0.05):

    distance = np.linalg.norm(np.array(gripper_position) - np.array(block_position))
    return distance <= threshold

# Detecting the blocks from the Vision sensor
def scanning_for_blocks(bricks, dummies):

    for angle in range(0, 360, 360):

        target_config = [deg_to_radian(angle), 0,0,0,0,0]

        # Robot moves
        move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, target_config)

        # Setting the robot tip
        previous_tip = sim.getObject('/Dummy')
        stack_height = 0.09

        # Sensor Time! (Ignored for now)
        # vision_data = sim.getVisionSensorImg(Vision_sensor, 0)
        # image = Image_retriver(vision_data, 256, 256)
        # blocks = detect_blocks(image)

        # for block in blocks:
        #     block_position = getBlock_Position(block)
        #     if block_position:
        #         detected_blocks.append(block_position)

        # Pick up the Block
        for brick, dummy in zip(bricks, dummies):

            if not bricks:
                break

            #gripper_position = sim.getObjectPosition(Robot_gripper, -1)
            brick_position = sim.getObjectPosition(dummy, -1)
            brick_orientation = sim.getObjectOrientation(dummy, -1)
            brick_matrix = sim.getObjectMatrix(brick, -1)
            
            print("Detected block to Pick UP OVER HEREREER")

            brick_position = [brick_position[0], brick_position[1], brick_position[2]+0.045]
            # = [brick_orientation[0], brick_orientation[1], brick_orientation[2]]

            sim.setObjectPose(previous_tip, -1, sim.getObjectPose(dummy, -1))

            # getting the block
            move_to_position(previous_tip, brick_position, brick_orientation)
                        
            sim.setObjectParent(brick, Robot_gripper, True)
            move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, target_config)

            # Change the target place to the plane
            target_dummy = sim.getObject('/TargetDummy')
            sim.setObjectPose(previous_tip, -1, sim.getObjectPose(target_dummy, -1))

            brick_position = sim.getObjectPosition(target_dummy, -1)
            brick_orientation = sim.getObjectOrientation(target_dummy, -1)

            # Move to the plane
            move_to_plane(previous_tip, brick_position, brick_orientation)

            # Increase the height of the target dummy to stack them - WRITE CODE HERE!!!
            stack_height += 0.02
            sim.setObjectPosition(target_dummy, -1, [brick_position[0], brick_position[1], stack_height])

            move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, target_config)

            # Log progress
            print("Block successfully picked and returned to default position!")
                    

# Setting up the RML Vectors (Rapid Motion Action) (velocity, acceleration, jerk)

# vel is Velocity:
# -> increasing it makes robot move faster across all the joints
# -> decreasing it makes robot move slower across all the joints
vel = 80

# accel is Acceleration
# -> increasing it makes robot reach max vel quickly
# -> decreasing it makes robot reach max vel slowly
accel = 100

# jerk is Jerk
# -> increasing it makes robot movement jerky
# -> decreasing it makes robot movement smooth
jerk = 20

# Getting Max RML
maxVel = [deg_to_radian(vel)] * 6
maxAccel = [deg_to_radian(accel)] * 6
maxJerk = [deg_to_radian(jerk)] * 6

def Robot_saying_HI():
    # The 3 Poses
    targetPos1 = [deg_to_radian(45),  deg_to_radian(0), deg_to_radian(90), deg_to_radian(90), 0, 0]
    targetPos2 = [deg_to_radian(45),  deg_to_radian(0), deg_to_radian(45), deg_to_radian(90), 0, 0]
    targetPos3 = [deg_to_radian(45),  deg_to_radian(0), deg_to_radian(90), deg_to_radian(90), 0, 0]

    # The movement function
    move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, targetPos1)
    time.sleep(0.05)
    move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, targetPos2)
    time.sleep(0.05)
    move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, targetPos3)

    print("The Robot said Hi!")

# Reading from Vision Sensor
def get_Vision_Data(Vision_sensor):
    # Getting the image from the vision sensor
    # 0 means RGB format of image
    try:
        result = sim.getVisionSensorImg(Vision_sensor, 0)
        if result:
            print("Vision sensor working!")
        else:
            print("Vision sensor not working!")
        return result
    except Exception as e:
        print(f"Error reading vision sensor: {e}")
        return -1

# Reading from Proximity Sensor
def get_Proximity_Data(Proximity_sensor):
    # Getting objects detected from the proximity sensor
    try:
        result = sim.readProximitySensor(Proximity_sensor)
        if result:
            print("Object detected by proximity sensor!")
        else:
            print("Proximity sensor not working!")
        return result
    except Exception as e:
        print(f"Error reading proximity sensor: {e}")
        return -1

# setting robot to default position
def robot_default_position():
    default_position = [0, 0, 0, 0, 0, 0]
    move_to_config(Robot_joints, maxVel, maxAccel, maxJerk, default_position)

# This is the main loop for the code
def main_loop(bricks, dummies):
    try:
        for _ in range(max_bricks):
            scanning_for_blocks(bricks, dummies)
            time.sleep(0.5)
    
    
    # To unexpectedly stop in case of infinite loop or error        
    except KeyboardInterrupt:
        print("Exiting loop.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Stopping simulation.")
        sim.stopSimulation()


# Calling the main function

# get link size
link1_size = 0.15
link2_size = 0.15

max_radius = 0.25
arm_position = sim.getObjectPosition(Robot_gripper, -1)

bricks = []
dummies = []
max_bricks = 1

for i in range(max_bricks):
    new_brick, new_dummy = place_bricks(Robot_base, arm_position, max_radius, bricks)
    bricks.append(new_brick)
    dummies.append(new_dummy)
    time.sleep(0.2)

robot_default_position()
main_loop(bricks,dummies)
#Robot_saying_HI()

# Stop simulation
sim.stopSimulation()
print("Simulation stopped.")
