#pragma once

#include <Eigen/Dense>
#include <functional>
#include <utility>
#include <vector>

namespace se3_efficient_distance {

/**
 * \brief Representation of a piecewise quadratic G-polynomial curve in SE(3).
 */
struct Se3PolynomialData {
    std::function<Eigen::Matrix4d(double)> curve;  ///< Evaluates C(s) for normalized s in [0, 1].
    std::vector<Eigen::Matrix4d> c_mats;           ///< Sampled control poses C_0, ..., C_K.
    std::vector<Eigen::Matrix4d> e1_mats;          ///< First-order se(3) coefficients E_{k,1}.
    std::vector<Eigen::Matrix4d> e2_mats;          ///< Second-order se(3) coefficients E_{k,2}.
    int k_segs = 0;                                ///< Number of polynomial segments K.
};

/**
 * \brief Finds all real roots of a cubic polynomial a*x^3 + b*x^2 + c*x + d = 0 using Cardano's formula.
 * \param a Cubic coefficient.
 * \param b Quadratic coefficient.
 * \param c Linear coefficient.
 * \param d Constant term.
 * \return Vector of real roots.
 */
std::vector<double> solve_cubic(double a, double b, double c, double d);

/**
 * \brief Minimizes the first-order approximate squared distance on a single G-polynomial segment.
 * \param h_target Query pose H in SE(3).
 * \param c_mat Segment origin pose C_k in SE(3).
 * \param e1_mat First-order Lie algebra coefficient E_{k,1}.
 * \param e2_mat Second-order Lie algebra coefficient E_{k,2}.
 * \param use_cardano If true, uses closed-form Cardano's formula; otherwise uses Eigen's polynomial solver.
 * \return Pair {s_local, squared_approx_dist} with s_local in [0, 1].
 */
std::pair<double, double> min_seg_approx(const Eigen::Matrix4d& h_target,
                                         const Eigen::Matrix4d& c_mat,
                                         const Eigen::Matrix4d& e1_mat,
                                         const Eigen::Matrix4d& e2_mat,
                                         bool use_cardano = true);

/**
 * \brief Constructs a C^1-continuous closed quadratic G-polynomial curve in SE(3) from sampled poses in O(K) time.
 * \param sample_points Sequence of K+1 poses [C_0, ..., C_K] in SE(3) with C_K = C_0 for a closed curve.
 * \return Interpolated G-polynomial curve structure.
 */
Se3PolynomialData create_se3_pol(const std::vector<Eigen::Matrix4d>& sample_points);

/**
 * \brief Samples a continuous parametric curve at K+1 uniform points and constructs its G-polynomial interpolant.
 * \param target_curve Parametric function mapping s in [0, 1] to SE(3).
 * \param k_segs Number of segments K.
 * \return Interpolated G-polynomial curve structure.
 */
Se3PolynomialData create_se3_pol(const std::function<Eigen::Matrix4d(double)>& target_curve, int k_segs);

}  // namespace se3_efficient_distance