import pyrealsense2 as rs

# Create a pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Enable the streams
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Function to list connected devices
def list_devices():
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No device connected")
    else:
        for dev in ctx.devices:
            print('Device connected:', dev.get_info(rs.camera_info.name), 'Serial:', dev.get_info(rs.camera_info.serial_number))

# List connected devices
list_devices()

# Try to start the pipeline
try:
    pipe.start(cfg)
    print("Pipeline started successfully")
except Exception as e:
    print(e)

# Add your frame processing code here
# while (True):
#     frame = pipe.wait_for_frames()
#     depth_frame = frame.get_depth_frame()
#     color_frame = frame.get_color_frame()
#     if not depth_frame or not color_frame:
#         continue
#     # Process frames here

# Stop the pipeline
pipe.stop()
