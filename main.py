import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf


# Extract the intrinsics parameter from the realsense D415 camera:
# Intrinsics extracted are the focal length, camera width and height,
#   distortion model, model coefficients, and two center projection points.
# The intrinsics are then used to compute the projection of a pixel into
#   3D space using pyrealsense deprojection method.
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    intrinsics = rs.intrinsics()

    # getting the width and height of camera
    intrinsics.width = cameraInfo.width
    intrinsics.height = cameraInfo.height

    # getting the projection points of the camera
    intrinsics.ppx = cameraInfo.ppx
    intrinsics.ppy = cameraInfo.ppy

    # getting the focal lengths of the camera
    intrinsics.fx = cameraInfo.fx
    intrinsics.fy = cameraInfo.fy

    # getting the distortion model of the camera
    intrinsics.model = cameraInfo.model

    # getting the coefficient of the calibration
    intrinsics.coeffs = cameraInfo.coeffs

    # pyrealsense's built-in deprojection method
    result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

    # returns the coordinates x, y, and z (forward, left, and up)
    return result[2], -result[0], -result[1]


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
print("Starting streaming")
cfg = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Fetch the profile of thd depth stream to get the camera intrinsics
profile = cfg.get_stream(rs.stream.depth)
intr = profile.as_video_stream_profile().get_intrinsics()

# Create an align object to align the color and depth frames
align = rs.align(rs.stream.color)

# Load pretrained weights
print("Loading model...")
PATH_TO_CKPT = "frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output is the boundingboxes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# The confidence scores and label of the object detected
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

print("Model loaded.")

colors_hash = {}

while True:

    frames = pipeline.wait_for_frames()

    # Align the color stream to the depth stream
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    scaled_size = (color_frame.width, color_frame.height)
    image_expanded = np.expand_dims(color_image, axis=0)

    # Perform inference using our TensorFlow session
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})

    # Squeeze the results into 2D and 1D arrays
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    # Iterate over the objects been detected
    for idx in range(int(num)):
        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]

        # Assign a random color to the bbox
        if class_ not in colors_hash:
            colors_hash[class_] = tuple(np.random.choice(range(256), size=3))

        # If confidence score is over 0.6, mark the object
        if score > 0.6:
            left = int(box[1] * color_frame.width)
            top = int(box[0] * color_frame.height)
            right = int(box[3] * color_frame.width)
            bottom = int(box[2] * color_frame.height)

            # The left top and right bottom points of the bbox
            p1 = (left, top)
            p2 = (right, bottom)

            # The center point of the bbox, where depth info will be further extracted
            point = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
            depth = depth_frame.get_distance(point[0], point[1])

            # Deproject the center pixel onto the 3D space
            x, y, z = convert_depth_to_phys_coord_using_realsense(point[0], point[1], depth, intr)

            # Limit the coordinates to 2 decimal points
            x = str(int(x * 100) / 100.0)
            y = str(int(y * 100) / 100.0)
            z = str(int(z * 100) / 100.0)

            # Draw boxes and print coordinates on screen
            r, g, b = colors_hash[class_]
            cv2.circle(color_image, point, 1, (int(r), int(g), int(b)), 3)
            cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)
            cv2.putText(color_image, "X: " + x + " Y: " + y + " Z: " + z,
                        (left + 10, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (int(r), int(g), int(b)), 2)

    # Quit if q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Show the image using OpenCV
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)

print("[INFO] stop streaming ...")
pipeline.stop()
