from nuscenes.nuscenes import NuScenes
import numpy as np
import pickle

# Load the Nuscenes dataset
dataroot = './data/nuscenes/'  # Replace with the correct path to your dataset
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

# Get all samples
all_samples = nusc.sample

def quaternion_to_rotation_matrix(q):
    """
    Converts quaternion [w, x, y, z] to a rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

# Loop through each sensor type
sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

data_annotations = []
for sample in all_samples:
    frame = {}
    for sensor in sensor_types:
        # Get the sensor data for the sample
        sensor_data_token = sample['data'][sensor]
        sensor_data = nusc.get('sample_data', sensor_data_token)

        # Get the calibration data
        calibrated_sensor = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])

        # Convert quaternion to rotation matrix
        sensor2lidar_rotation = quaternion_to_rotation_matrix(calibrated_sensor['rotation'])

        # Prepare the calibration info for this sensor
        calibration_info = {
            'data_path': f"./data/nuscenes/{sensor_data['filename']}",
            'type': sensor,
            'sample_data_token': sensor_data_token,
            'sensor2ego_translation': np.array(calibrated_sensor['translation']),
            'sensor2ego_rotation': np.array(calibrated_sensor['rotation']),
            'ego2global_translation': np.array(ego_pose['translation']),
            'ego2global_rotation': np.array(ego_pose['rotation']),
            'timestamp': sensor_data['timestamp'],
            'sensor2lidar_rotation': sensor2lidar_rotation,
            'sensor2lidar_translation': np.array(calibrated_sensor['translation']),
            'cam_intrinsic': np.array(calibrated_sensor['camera_intrinsic'])
        }
        
        frame[sensor] = calibration_info
    data_annotations.append(frame)

pickle.dump(data_annotations, open('nuscenes_infos.pkl', 'wb'))