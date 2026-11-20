#pragma once

#include <Eigen/Dense>

namespace se3_efficient_distance {

/**
 * \brief Maps a 3D vector to a 3x3 skew-symmetric matrix in so(3).
 * \param w Angular velocity or rotation vector in R^3.
 * \return 3x3 skew-symmetric matrix.
 */
Eigen::Matrix3d skew(const Eigen::Vector3d& w);

/**
 * \brief Computes the analytical inverse of a rigid-body transformation in SE(3).
 * \param h Homogeneous transformation matrix in SE(3).
 * \return Inverse transformation matrix H^{-1} in SE(3).
 */
Eigen::Matrix4d inv_se3(const Eigen::Matrix4d& h);

/**
 * \brief Extracts the 3D vector from a 3x3 skew-symmetric matrix in so(3).
 * \param w_mat Skew-symmetric matrix in so(3).
 * \return Corresponding vector in R^3.
 */
Eigen::Vector3d vee_so3(const Eigen::Matrix3d& w_mat);

/**
 * \brief Computes the exponential map from so(3) to SO(3) via Rodrigues' formula.
 * \param phi Rotation vector in R^3.
 * \return Rotation matrix in SO(3).
 */
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& phi);

/**
 * \brief Computes the principal logarithm from SO(3) to so(3).
 * \param r Rotation matrix in SO(3).
 * \return Axis-angle rotation vector in R^3.
 */
Eigen::Vector3d log_so3(const Eigen::Matrix3d& r);

/**
 * \brief Computes the left Jacobian of SO(3).
 * \param phi Rotation vector in R^3.
 * \return 3x3 left Jacobian matrix.
 */
Eigen::Matrix3d jac_left_so3(const Eigen::Vector3d& phi);

/**
 * \brief Computes the inverse of the left Jacobian of SO(3).
 * \param phi Rotation vector in R^3.
 * \return 3x3 inverse left Jacobian matrix.
 */
Eigen::Matrix3d inv_jac_left_so3(const Eigen::Vector3d& phi);

/**
 * \brief Computes the exponential map from the Lie algebra se(3) to SE(3).
 * \param a 4x4 twist matrix in se(3).
 * \return Homogeneous transformation matrix in SE(3).
 */
Eigen::Matrix4d exp_se3(const Eigen::Matrix4d& a);

/**
 * \brief Computes the principal logarithm from SE(3) to se(3).
 * \param h Homogeneous transformation matrix in SE(3).
 * \return 4x4 twist matrix in se(3).
 */
Eigen::Matrix4d log_se3(const Eigen::Matrix4d& h);

/**
 * \brief Vectorizes a 4x4 matrix in se(3) into a 6D twist vector [v; w].
 * \param a 4x4 matrix in se(3).
 * \return 6D vector [a_x, a_y, a_z, alpha_x, alpha_y, alpha_z]^T.
 */
Eigen::Matrix<double, 6, 1> vec(const Eigen::Matrix4d& a);

/**
 * \brief Maps a 6D twist vector [v; w] to its 4x4 matrix representation in se(3).
 * \param a 6D twist vector.
 * \return 4x4 matrix in se(3).
 */
Eigen::Matrix4d vec_inv(const Eigen::Matrix<double, 6, 1>& a);

/**
 * \brief Computes the 6x6 left Jacobian J of SE(3).
 * \param a 4x4 matrix in se(3).
 * \return 6x6 left Jacobian matrix.
 */
Eigen::Matrix<double, 6, 6> left_jac(const Eigen::Matrix4d& a);

/**
 * \brief Computes the matrix representation I_A of exp(-A) D exp_A in SE(3).
 * \param a 4x4 matrix in se(3).
 * \return 6x6 matrix I_A.
 */
Eigen::Matrix<double, 6, 6> i_se3(const Eigen::Matrix4d& a);

/**
 * \brief Evaluates the linear operator L_A[E] = vec^{-1}(I_{-A}^{-1} vec(E)).
 * \param a Base point 4x4 matrix in se(3).
 * \param e Direction 4x4 matrix in se(3).
 * \param i1_inv Optional cached 3x3 upper-left inverse block of I_{-A}.
 * \param b_inv Optional cached 3x3 off-diagonal inverse block of I_{-A}.
 * \return 4x4 matrix L_A[E] in se(3).
 */
Eigen::Matrix4d l_se3(const Eigen::Matrix4d& a,
                      const Eigen::Matrix4d& e,
                      Eigen::Matrix3d* i1_inv = nullptr,
                      Eigen::Matrix3d* b_inv = nullptr);

}  // namespace se3_efficient_distance