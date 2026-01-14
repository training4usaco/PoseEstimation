import os
import numpy as np
import matplotlib.pyplot as plt

FOCAL_FILE = 'Pose/focal.txt'
POSES_FILE = 'Pose/poses.txt'
OUTPUT_DIR = 'projections'

dir = os.path.dirname(os.path.abspath(__file__))

focal_path = os.path.join(dir, FOCAL_FILE)
poses_path = os.path.join(dir, POSES_FILE)
camera_coordinates = []
joint_coordinates = []

LEFT_BONES = [
    (4, 5), (5, 6), (7, 8), (8, 9), (9, 10)
]

RIGHT_BONES = [
    (1, 2), (2, 3), (7, 11), (11, 12), (12, 13)
]

CORE_BONES = [
    (0, 1), (0, 4), (0, 7)
]

BONES = [
'Hip',
'Right Hip',
'Right Knee',
'Right Ankle',
'Left Hip',
'Left Knee',
'Left Ankle',
'Neck',
'Left Upper Arm',
'Left Elbow',
'Left Wrist',
'Right Upper Arm',
'Right Elbow',
'Right Wrist'
]

TITLES = [
'Hip',
'RHip',
'RKnee',
'RAnkle',
'LHip',
'LKnee',
'LAnkle',
'Neck',
'LUpperArm',
'LElbow',
'LWrist',
'RUpperArm',
'RElbow',
'RWrist'
]


with open(FOCAL_FILE, 'r') as f:
    focal_mm = float(f.read().strip())

with open(POSES_FILE, 'r') as f:
    data = f.read().splitlines()
    for pose in data:
        coordinates = pose.split()
        camera_coordinates.append([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])
        curr_coordinates = []
        for i in range(3, 45, 3):
            curr_coordinates.append([float(coordinates[i]), float(coordinates[i + 1]), float(coordinates[i + 2])])
        joint_coordinates.append(curr_coordinates)

output_csv = []
r_headers = ["R_00", "R_01", "R_02", "R_10", "R_11", "R_12", "R_20", "R_21", "R_22"]
j_headers = [f"{name}_x,{name}_y" for name in TITLES]

header = "Frame," + ",".join(r_headers) + "," + ",".join(j_headers)

format_list = ['%d'] + ['%.15f'] * 37

final_rows = []

for i in range(20):
    n_hat = np.array(joint_coordinates[i][0]) - np.array(camera_coordinates[i])
    n_hat /= np.linalg.norm(n_hat)

    k = np.array([0, 0, 1])
    u_hat = np.cross(n_hat, k)
    if np.linalg.norm(u_hat) < 1e-6:
        u_hat = np.array([1, 0, 0])
    else:
        u_hat /= np.linalg.norm(u_hat)

    v_hat = np.cross(u_hat, n_hat)
    v_hat /= np.linalg.norm(v_hat)

    R = np.array([u_hat, v_hat, n_hat])
    print("Camera Orientation:")
    print(R)

    projected_coordinates = []
    for j in range(14):
        vec = np.array(joint_coordinates[i][j]) - np.array(camera_coordinates[i])
        x_prime = np.dot(vec, u_hat)
        y_prime = np.dot(vec, v_hat)
        z_prime = np.dot(vec, n_hat)

        if z_prime < 1e-6:
            z_prime = 1e-6

        x_double_prime = focal_mm * (x_prime / z_prime)
        y_double_prime = focal_mm * (y_prime / z_prime)
        projected_coordinates.append([float(x_double_prime), float(y_double_prime)])
    print(f"Projected Coordinates {i}:")
    print(projected_coordinates)

    frame_id = i
    coords = [item for pair in projected_coordinates for item in pair]
    R_flat = np.array([u_hat, v_hat, n_hat]).flatten()
    row = np.concatenate(([frame_id], R_flat, coords))
    final_rows.append(row)

    for j in range(14):
        print(f"{BONES[j]} & ({projected_coordinates[j][0]: 0.15f}, {projected_coordinates[j][1]: 0.15f}) \\\\")

    x = [coordinate[0] for coordinate in projected_coordinates]
    y = [coordinate[1] for coordinate in projected_coordinates]

    # Normalization so we can have nice, square images that are centered at the hip joint
    image_dimension = max(max(max(x), abs(min(x))), max(max(y), abs(min(y))))
    x = [val / image_dimension for val in x]
    y = [val / image_dimension for val in y]
    projected_coordinates = [[coordinate[0] / image_dimension, coordinate[1] / image_dimension] for coordinate in projected_coordinates]

    fig, ax = plt.subplots(figsize = (8, 6))
    for joint_a, joint_b in LEFT_BONES:
        if joint_a < len(projected_coordinates) and joint_b < len(projected_coordinates):
            xa, ya = projected_coordinates[joint_a]
            xb, yb = projected_coordinates[joint_b]
            ax.plot([xa, xb], [ya, yb], c = 'red', linewidth = 2, zorder = 1)

    for joint_a, joint_b in RIGHT_BONES:
        if joint_a < len(projected_coordinates) and joint_b < len(projected_coordinates):
            xa, ya = projected_coordinates[joint_a]
            xb, yb = projected_coordinates[joint_b]
            ax.plot([xa, xb], [ya, yb], c = 'blue', linewidth = 2, zorder = 1)

    for joint_a, joint_b in CORE_BONES:
        if joint_a < len(projected_coordinates) and joint_b < len(projected_coordinates):
            xa, ya = projected_coordinates[joint_a]
            xb, yb = projected_coordinates[joint_b]
            ax.plot([xa, xb], [ya, yb], c = 'black', linewidth = 2, zorder = 1)

    ax.scatter(x, y, c = 'black', s = 30, zorder = 2)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    plt.axis('off')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, f"pose_{i:02d}.png")
    plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

OUTPUT_FILE = "full_pose_data.csv"
np.savetxt(OUTPUT_FILE, final_rows, delimiter = ",", header = header, comments = '', fmt = format_list)