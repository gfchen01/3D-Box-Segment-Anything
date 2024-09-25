from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import numpy as np
import pickle

from pyquaternion import Quaternion

# Load the Nuscenes dataset
dataroot = './data/nuscenes/'  # Replace with the correct path to your dataset
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

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
    
def rotation_matrix_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion [w, x, y, z].
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    
    w = np.sqrt(1 + r11 + r22 + r33) / 2
    x = (r32 - r23) / (4 * w)
    y = (r13 - r31) / (4 * w)
    z = (r21 - r12) / (4 * w)
    
    return np.array([w, x, y, z])

# Loop through each sensor type
sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# data_annotations_train = []
# data_annotations_val = []
data_annotations = []
for sample in all_samples:
    frame = {}
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    # reference: https://www.nuscenes.org/nuscenes 
    lidar_to_ego_R = quaternion_to_rotation_matrix(lidar_calib['rotation'])
    lidar_to_ego_t = np.array(lidar_calib['translation'])
    
    frame['LIDAR_TOP'] = {
        'data_path': f"data/nuscenes/{lidar_data['filename']}",
        'type': 'LIDAR_TOP',
        'sample_data_token': lidar_token
    }

    for sensor in sensor_types:
        # Get the sensor data for the sample
        sensor_data_token = sample['data'][sensor]
        sensor_data = nusc.get('sample_data', sensor_data_token)

        # Get the calibration data
        ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
        ego2global_R = quaternion_to_rotation_matrix(ego_pose['rotation'])
        ego2global_t = np.array(ego_pose['translation'])
        
        calibrated_sensor = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
        sensor_to_ego_R = quaternion_to_rotation_matrix(calibrated_sensor['rotation'])
        sensor_to_ego_t = np.array(calibrated_sensor['translation'])
        
        sensor_to_lidar_R = lidar_to_ego_R.T @ sensor_to_ego_R
        sensor_to_lidar_t = lidar_to_ego_R.T @ (sensor_to_ego_t - lidar_to_ego_t)

        # Prepare the calibration info for this sensor
        calibration_info = {
            'data_path': f"data/nuscenes/{sensor_data['filename']}",
            'type': sensor,
            'sample_data_token': sensor_data_token,
            'sensor2ego_translation': sensor_to_ego_t,
            'sensor2ego_rotation': rotation_matrix_to_quaternion(sensor_to_ego_R),
            'ego2global_translation': ego2global_t,
            'ego2global_rotation': rotation_matrix_to_quaternion(ego2global_R),
            'timestamp': sensor_data['timestamp'],
            'sensor2lidar_rotation': sensor_to_lidar_R,
            'sensor2lidar_translation': sensor_to_lidar_t,
            'cam_intrinsic': np.array(calibrated_sensor['camera_intrinsic'])
        }
        
        frame[sensor] = calibration_info
    
    # if sample['scene_token'] in splits.mini_train:
    #     data_annotations_train.append(frame)    
    # elif sample['scene_token'] in splits.mini_val:
    #     data_annotations_val.append(frame)
    # else:
    #     print("Invalid split!")
    #     break
    data_annotations.append(frame)
    
# pickle.dump(data_annotations_train, open('nuscenes_camera_infos_train.pkl', 'wb'))
# pickle.dump(data_annotations_val, open('nuscenes_camera_infos_val.pkl', 'wb'))

pickle.dump(data_annotations, open('nuscenes_camera_infos.pkl', 'wb'))

# nusc.render_sample(all_samples[0]['token'])