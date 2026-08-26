ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
# ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
TASK = "task1"  # "task1" or "task2"; overridden by argv
ROBOT_SCENE = "../tasks/task1_pick_place/" + TASK + ".xml"
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo0" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Pelvis is welded in task1; skip elastic-band glfw callback

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
