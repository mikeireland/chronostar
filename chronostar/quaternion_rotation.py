import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_matrix(quaternion):
    """
    Given a quaternion, generate its rotation matrix. Only unit quaternion
    represent rotation. Non-normalised quaternions will be normalised.

    :param quaternion: 4 element vector: scalar first quaternion notation
    :return: ndarray - 6x6 rotation matrix corresponding to the
        quaternion and its transpose
    """
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    s = np.linalg.norm([r, i, j, k])**(-2)
    rot_mat_3d = np.array([
        [1-2*s*(j**2 + k**2), 2*s*(i*j - k*r),     2*s*(i*k + j*r)],
        [2*s*(i*j + k*r),     1-2*s*(i**2 + k**2), 2*s*(j*k - i*r)],
        [2*s*(i*k - j*r),     2*s*(j*k + i*r),     1-2*s*(i**2 + j**2)]
    ])
    rot_mat = np.block([
        [rot_mat_3d,       np.zeros((3, 3))],
        [np.zeros((3, 3)), rot_mat_3d]
    ])
    rot_mat_transpose = np.transpose(rot_mat)
    # print("6D rotation matrix: \n", rot_mat)
    # print("6D rotation matrix transpose: \n", rot_mat_transpose)

    return rot_mat, rot_mat_transpose


def scale_covariance(sigma_x, sigma_y_z, sigma_u, sigma_v_w):
    """
    Scale parameters of the covariance matrix to reduce the total number of parameters.
    Standard deviation in y,z direction is scaled to 1 and the remaining parameters are
    scaled accordingly.

    :param sigma_x: standard deviation in x - direction
    :param sigma_y_z: standard deviation in y,z - direction
    :param sigma_u: standard deviation in u - velocity axis
    :param sigma_v_w: standard deviation in v,w - velocity axis
    :return: float - scaled parameters
    """
    alpha = sigma_x/sigma_y_z
    beta = sigma_u/sigma_y_z
    gamma = sigma_v_w/sigma_y_z
    return alpha, beta, gamma


def rotate(alpha, beta, gamma, cov_x_v, quaternion):
    """
    Perform rotation on an ellipsoid using quaternion rotation

    :param alpha: scaled x-direction standard deviation
    :param gamma: u - velocity axis
    :param beta: scaled v,w - velocity axis
    :param cov_x_v: covariance of x-axis position and y axis velocity
    :param quaternion: rotation quaternion
    :return: ndarray - 6x6 rotated covariance matrix
    """
    mag_q = np.linalg.norm(quaternion)
    if abs(mag_q-1) > 1e-10:
        quaternion/mag_q
    rot_mat, rot_mat_transpose = rotation_matrix(quaternion)
    cov_mat = np.array([
        [alpha**2, 0., 0., 0.,       cov_x_v, 0.],
        [0.,       1., 0., 0.,       0.,      0.],
        [0.,       0., 1., 0.,       0.,      0.],
        [0.,       0., 0., beta**2, 0.,      0.],
        [cov_x_v,  0., 0., 0.,       gamma**2, 0.],
        [0.,       0., 0., 0.,       0.,      gamma**2]
    ])
    return np.matmul(rot_mat, np.matmul(cov_mat, rot_mat_transpose))


def to_euler(quaternion):
    """

    :param quaternion: Quaternion to be converted
    :return: [x-axis, y-axis, z-axis] rotation
    """
    quaternion = quat_normalise(quaternion)
    q_r = quaternion[0]
    q_i = quaternion[1]
    q_j = quaternion[2]
    q_k = quaternion[3]

    # roll(x - axis rotation)
    sin_cos_rp = 2 * (q_r * q_i + q_j * q_k)
    cos_cos_rp = 1 - 2 * (q_i**2 + q_j**2)
    roll = np.math.atan2(sin_cos_rp, cos_cos_rp)

    # pitch (y - axis rotation)
    sin_p = 2 * (q_r * q_j - q_k * q_i)
    if abs(sin_p) >= 1:
        pitch = np.math.copysign(np.pi/2, sin_p)  # use 90 degrees if out of range
    else:
        pitch = np.math.asin(sin_p)

    # yaw(z - axis rotation)
    sin_cos_yp = 2 * (q_r * q_k + q_i * q_j)
    cos_cos_yp = 1 - 2 * (q_j * q_j + q_k * q_k)
    yaw = np.math.atan2(sin_cos_yp, cos_cos_yp)

    return roll, pitch, yaw


def to_quat(roll, pitch, yaw):
    """

    :param roll: x - axis rotation
    :param pitch: y - axis rotation
    :param yaw: z - axis rotation
    :return: Quaternion generated from the Euler angles
    """
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)

    q_r = cr * cp * cy + sr * sp * sy
    q_i = sr * cp * cy - cr * sp * sy
    q_j = cr * sp * cy + sr * cp * sy
    q_k = cr * cp * sy - sr * sp * cy

    return np.array([q_r, q_i, q_j, q_k])


def quat_normalise(quaternion):
    return quaternion/np.linalg.norm(quaternion)


# def non_norm_to_euler(quaternion):
#     q_r = quaternion[0]
#     q_i = quaternion[1]
#     q_j = quaternion[2]
#     q_k = quaternion[3]
#
#     sqw = q_r * q_r
#     sqx = q_i * q_i
#     sqy = q_j * q_j
#     sqz = q_k * q_k
#     unit = sqx + sqy + sqz + sqw  # if normalised is one, otherwise is correction factor
#     test = q_i * q_j + q_k * q_r
#     if test > 0.499 * unit:  # singularity at north pole
#         heading = 2 * np.math.atan2(q_i, q_r)
#         attitude = np.pi / 2
#         bank = 0
#         return [heading, attitude, bank]
#     if test < -0.499 * unit:  # singularity at south pole
#         heading = -2 * np.math.atan2(q_i, q_r)
#         attitude = -np.pi/2
#         bank = 0
#         return [heading, attitude, bank]
#
#     heading = np.math.atan2(2 * q_j * q_r - 2 * q_i * q_k, sqx - sqy - sqz + sqw)
#     attitude = np.math.asin(2 * test / unit)
#     bank = np.math.atan2(2 * q_i * q_r - 2 * q_j * q_k, -sqx + sqy - sqz + sqw)
#     return [heading, attitude, bank]


# q1 = np.array([1, 2, 3, 4])
# q2 = quat_normalise(q1)
# print("Quaternion:", q1)
# print("Normalised quaternion:", q2)
# euler = to_euler(q1)
# euler2 = to_euler(q2)
# print("Quaternion original -> Euler:", euler)
# print("Quaternion normalised -> Euler:", euler2)
# print("Quaternion original -> Euler -> quaternion:", to_quat(euler[0], euler[1], euler[2]))
# print("Quaternion normalised -> Euler -> quaternion:", to_quat(euler2[0], euler2[1], euler2[2]))
# # rot_mat_ex, rot_mat_transpose_ex = rotation_matrix(q1)
# # rot_mat2_ex, rot_mat_transpose2_ex = rotation_matrix(q2)
# # print("6D rotation matrix: \n", rot_mat_ex)
# # print("6D rotation matrix normalised: \n", rot_mat2_ex)
# # print("Quaternion original -> non_norm Euler:", non_norm_to_euler(q1))
# # print("Quaternion normalised -> non_norm Euler:", non_norm_to_euler(q2))
#
# a = R.from_quat([2, 3, 4, 1])
# b = R.from_quat([0.36514837, 0.54772256, 0.73029674, 0.18257419])
# print(a.as_euler('xyz', degrees=False))
# print(b.as_euler('xyz', degrees=False))
# c = R.from_euler('xyz', [1.42889928, -0.3398369, 2.35619448], degrees=False)
# print(c.as_quat())
