import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

PI = np.pi

#! modified DH params
#! pelvis_pitch, lower_waist_pitch, lower_waist_yaw,
#! upper_waist_yaw, upper_waist_pitch, neck_yaw, neck_pitch
#! 机器鼠右视图， 俯仰角对应坐标系，Z轴向内
# [alpha_{i}, a_{i}, d_{i+1}, theta_{i+1}]
# i:0,1,...,7
# dh_params = np.array(
#     [
#         [-PI / 2, 0, 0, -0.33],
#         [0, 39, 0, 0.33],
#         [PI / 2, 0, 0, 0],
#         [0, 40, 0, 0],
#         [-PI / 2, 0, 0, 0],
#         [PI / 2, 55, 0, 0],
#         [-PI / 2, 12, 0, 0.25],
#         [0, 43.5, 0, 0],
#     ]
# )

# offsets = np.array(
#     [
#         [-23, 0, 0],
#         [-5, -25, 0],
#         [0, 0, 25],
#         [0, 0, 25],
#         [-18, 0, 1],
#         [-22, -22, 0],
#         [33, -15, 0],
#     ]
# )

dh_params = np.array(
    [
        [-PI / 2, 0, 0, -0.33],
        [0, 0.039, 0, 0.33],
        [PI / 2, 0, 0, 0],
        [0, 0.040, 0, 0],
        [-PI / 2, 0, 0, 0],
        [PI / 2, 0.055, 0, 0],
        [-PI / 2, 0.012, 0, 0.25],
        [0, 0.0435, 0, 0],
    ]
)

offsets = (
    np.array(
        [
            [-23, 0, 0],
            [-5, -25, 0],
            [0, 0, 25],
            [0, 0, 25],
            [-18, 0, 10],
            [-22, -22, 0],
            [33, -15, 0],
        ]
    )
    / 1000
)

offsets_joints_index = np.array([0, 1, 2, 3, 5, 6, 6], dtype=int)


def show_joint_positions(joint_points, spine_points, outline_points, show: bool = True):
    def show_position(points, color_point="r", color_line="b"):
        num_frame = points.shape[0]
        for i_frame in range(num_frame):
            x = points[i_frame, :, 0]
            y = points[i_frame, :, 1]
            z = points[i_frame, :, 2]
            ax.scatter(x, y, z, c=color_point, marker="o")
            ax.plot(x, y, z, c=color_line)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if joint_points is not None:
        if joint_points.ndim == 2:
            joint_points = joint_points.reshape(1, -1, 3)
        show_position(joint_points, color_point="green", color_line="cyan")
    if spine_points is not None:
        if spine_points.ndim == 2:
            spine_points = spine_points.reshape(1, -1, 3)
        show_position(spine_points, color_point="purple", color_line="blue")
    if outline_points is not None:
        if outline_points.ndim == 2:
            outline_points = outline_points.reshape(1, -1, 3)
        show_position(outline_points, color_point="yellow", color_line="orange")

    ax.axis("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Joint Postions")

    if show:
        plt.show()


def show_joint_positions_compare(
    joint_points_1: np.array,
    spine_points_1: np.array,
    outline_points_1: np.array,
    joint_points_2: np.array,
    spine_points_2: np.array,
    outline_points_2: np.array,
    show: bool = True,
):
    def show_position(points, color_point="r", color_line="b", linewidth=3):
        num_frame = points.shape[0]
        for i_frame in range(num_frame):
            x = points[i_frame, :, 0]
            y = points[i_frame, :, 1]
            z = points[i_frame, :, 2]
            ax.scatter(x, y, z, c=color_point, marker="o", linewidth=linewidth)
            ax.plot(x, y, z, c=color_line, linewidth=linewidth)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Group1
    if joint_points_1 is not None:
        if joint_points_1.ndim == 1:
            joint_points_1 = joint_points_1.reshape(1, -1)
        show_position(joint_points_1, color_point="peru", color_line="saddlebrown")
    if spine_points_1 is not None:
        if spine_points_1.ndim == 2:
            spine_points_1 = spine_points_1.reshape(1, -1, 3)
        show_position(spine_points_1, color_point="darkorange", color_line="orange")
    if outline_points_1 is not None:
        if outline_points_1.ndim == 2:
            outline_points_1 = outline_points_1.reshape(1, -1, 3)
        show_position(outline_points_1, color_point="brown", color_line="lightcoral")
    # Group2
    if joint_points_2 is not None:
        if joint_points_2.ndim == 1:
            joint_points_2 = joint_points_2.reshape(1, -1)
        show_position(joint_points_2, color_point="green", color_line="lightgreen")
    if spine_points_2 is not None:
        if spine_points_2.ndim == 2:
            spine_points_2 = spine_points_2.reshape(1, -1, 3)
        show_position(spine_points_2, color_point="purple", color_line="cyan")
    if outline_points_2 is not None:
        if outline_points_2.ndim == 2:
            outline_points_2 = outline_points_2.reshape(1, -1, 3)
        show_position(outline_points_2, color_point="navy", color_line="blue")

    ax.axis("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Joint Postions")

    if show:
        plt.show()


def trans_joints_format(joint_points: np.array) -> np.array:
    if joint_points.ndim == 1:
        joint_points = joint_points.reshape(1, -1)

    num_frames = joint_points.shape[0]
    num_points = joint_points.shape[1] // 3

    joint_points_result = np.zeros(shape=(num_frames, num_points, 3))

    for i_frame in range(num_frames):
        for i_point in range(num_points):
            joint_points_result[i_frame, i_point, :] = joint_points[
                i_frame, i_point * 3 : i_point * 3 + 3
            ]

    return joint_points_result


def get_ModifiedDH_T(alpha: float, a: float, d: float, theta: float) -> np.array:
    T_rot1 = t3d.affines.compose(
        [0, 0, 0], t3d.euler.euler2mat(alpha, 0, 0, "sxyz"), [1, 1, 1]
    )
    T_trans = t3d.affines.compose(
        [a, 0, d], t3d.euler.euler2mat(0, 0, 0, "sxyz"), [1, 1, 1]
    )
    T_rot2 = t3d.affines.compose(
        [0, 0, 0], t3d.euler.euler2mat(0, 0, theta, "sxyz"), [1, 1, 1]
    )
    return T_rot1 @ T_trans @ T_rot2


def get_joint_angle_all(
    joint_angles: np.ndarray,
    inversion: bool = True,
    degree: bool = True,
    head: bool = True,
    pitch_inversion: bool = False,
) -> np.array:
    #! 批处理
    if isinstance(joint_angles, list):
        joint_angles = np.array(joint_angles)

    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)

    num_frames = joint_angles.shape[0]
    num_angles = joint_angles.shape[1]

    if head:
        joint_angles_all = np.zeros(shape=(num_frames, 8))
    else:
        joint_angles_all = np.zeros(shape=(num_frames, 7))

    if num_angles == 5:
        joint_angles_all[:, 0:3] = joint_angles[:, 0:3]
        joint_angles_all[:, 3] = joint_angles[:, 2]
        joint_angles_all[:, 4] = joint_angles[:, 1]
        joint_angles_all[:, 5:7] = joint_angles[:, 3:5]

    elif num_angles == 7:
        joint_angles_all[:, :7] = joint_angles
    else:
        raise ValueError("The length of command should be 5 or 7!")

    if degree:
        joint_angles_all = joint_angles_all * np.pi / 180
    if inversion:
        joint_angles_all = -joint_angles_all
    if pitch_inversion:
        joint_angles_all[:, 0:2] = -joint_angles_all[:, 0:2]
        joint_angles_all[:, 4] = -joint_angles_all[:, 4]
        joint_angles_all[:, 6] = -joint_angles_all[:, 6]

    return joint_angles_all


def get_joint_points(dh_params: np.array, joint_num_all: int, joint_angles: np.array):
    #! 单一处理
    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)
    if joint_angles.shape[0] == 1:
        joint_angles = joint_angles[0, :]

    joint_points = np.zeros(shape=(1, joint_num_all, 3))
    T_joints = np.zeros(shape=(joint_num_all, 4, 4))

    for i_joint in range(joint_num_all):
        alpha, a, d, theta = dh_params[i_joint]
        theta += joint_angles[i_joint]

        T_temp = get_ModifiedDH_T(alpha, a, d, theta)

        if i_joint == 0:
            T_joints[i_joint] = np.eye(4) @ T_temp
        else:
            T_joints[i_joint] = T_joints[i_joint - 1] @ T_temp

        joint_points[0, i_joint, :] = T_joints[i_joint, 0:3, 3]

    return joint_points, T_joints


def get_spine_points(
    offsets: np.array, offsets_joints_index: np.array, T_joints: np.array
):
    #! 单一处理
    def get_spine_position(offset: np.array, T_joint: np.array):
        T_trans = t3d.affines.compose(
            offset, t3d.euler.euler2mat(0, 0, 0, "sxyz"), [1, 1, 1]
        )
        return (T_joint @ T_trans)[0:3, 3]

    spine_num = offsets.shape[0]
    spine_points = np.zeros(shape=(1, spine_num, 3))

    for i_spine in range(spine_num):
        if i_spine == 0:
            spine_points[0, i_spine, :] = get_spine_position(
                offsets[i_spine], np.eye(4)
            )
        else:
            spine_points[0, i_spine, :] = get_spine_position(
                offsets[i_spine], T_joints[offsets_joints_index[i_spine]]
            )

    return spine_points


def get_outline_points(points: np.array, num_interp_points: int) -> np.array:
    #! 批处理
    if points.ndim == 2:
        points = points.reshape(1, -1, 3)

    num_frames = points.shape[0]
    outline_points = np.zeros(shape=(num_frames, num_interp_points, 3))
    n = np.arange(points.shape[1])
    n_interp = np.linspace(n[0], n[-1], num_interp_points)

    for i_frame in range(num_frames):
        points_ = points[i_frame]
        interp = PchipInterpolator(n, points_.T, axis=1, extrapolate=True)
        points_interp = interp(n_interp)

        for i in range(3):
            outline_points[i_frame, :, i] = points_interp[i, :]

    return outline_points


def get_outline_points_from_angle(
    joint_angles: np.ndarray,
    num_interp_points: int = 100,
    dh_params: np.array = dh_params,
    offsets: np.array = offsets,
    offsets_joints_index: np.array = offsets_joints_index,
    inversion: bool = False,
    pitch_inversion: bool = False,
    degree: bool = False,
    head: bool = True,
) -> np.array:
    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)
    joint_num_all = int(8) if head is True else 7  #!TEMP
    num_frames = joint_angles.shape[0]
    num_spine = offsets.shape[0]

    # joint_points = np.zeros(shape=(num_frames, num_spine, 3))
    joint_points = np.zeros(shape=(num_frames, joint_num_all, 3))  #!Temp
    spine_points = np.zeros(shape=(num_frames, num_spine, 3))
    outline_points = np.zeros(shape=(num_frames, num_interp_points, 3))
    joint_angles_all = get_joint_angle_all(
        joint_angles, inversion, degree, head, pitch_inversion
    )

    for i_frame in range(num_frames):
        T_joints_temp = np.zeros(shape=(joint_num_all, 4, 4))
        joint_points[i_frame, :, :], T_joints_temp = get_joint_points(
            dh_params=dh_params,
            joint_num_all=joint_num_all,
            joint_angles=joint_angles_all[i_frame, :],
        )
        spine_points[i_frame, :, :] = get_spine_points(
            offsets=offsets,
            offsets_joints_index=offsets_joints_index,
            T_joints=T_joints_temp,
        )

    outline_points = get_outline_points(
        points=spine_points, num_interp_points=num_interp_points
    )

    return joint_points, spine_points, outline_points


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=5, suppress=True)

    # Args
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.joint_num_movable = 7
    args.joint_num_all = 8  # 包含鼻子顶端
    args.joint_num_active = 5
    args.joint_num_passive = 2

    #! 俯 俯 偏 偏 俯

    # joint_angles = get_joint_angle_all([10, 10, 0, 0, 10], inversion=True, degree=True)
    joint_angles = get_joint_angle_all([0, 0, 30, 30, 0], inversion=True, degree=True)
    joint_points, T_joints = get_joint_points(
        dh_params, args.joint_num_all, joint_angles
    )
    spine_points = get_spine_points(
        offsets=offsets,
        offsets_joints_index=offsets_joints_index,
        T_joints=T_joints,
    )
    # print(spine_points)
    outline_points = get_outline_points(points=spine_points, num_interp_points=100)
    # print(outline_points)

    show_joint_positions(
        joint_points=joint_points,
        spine_points=spine_points,
        outline_points=outline_points,
        show=False,
    )

    angs = np.array([[0, 0, 0, 0, 0], [0, 30, 0, 0, 0], [0, 30, 30, 0, 0]])
    _, _, outline_points = get_outline_points_from_angle(angs, 100)
    print(type(outline_points))
    print(outline_points.shape)
    show_joint_positions(
        joint_points=None, spine_points=None, outline_points=outline_points, show=True
    )
