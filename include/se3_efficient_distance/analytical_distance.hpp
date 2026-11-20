#pragma once

#include "se3_efficient_distance/polynomial_curves.hpp"

#include <Eigen/Dense>

namespace se3_efficient_distance {

/**
 * \brief Result of the point-to-curve distance query in SE(3).
 */
struct DistanceResult {
    double s_star = 0.0;                                       ///< Optimal normalized curve parameter in [0, 1].
    double distance = 0.0;                                     ///< Lie group geodesic distance ||log(H^{-1} C(s*))||_F.
    Eigen::Matrix4d closest_pose = Eigen::Matrix4d::Identity(); ///< Closest pose C(s*) on the G-polynomial curve.
};

/**
 * \brief Computes the approximate distance and closest point from a pose H to a G-polynomial curve (Algorithm 1).
 *
 * Solves the cubic polynomial root-finding problem on each segment to obtain an initial parameter estimate,
 * followed by a local coordinate descent refinement on the exact SE(3) distance function.
 *
 * \param h_target Query pose H in SE(3).
 * \param poly_data Precomputed G-polynomial curve representation.
 * \param use_cardano If true, uses closed-form Cardano's formula for cubic roots.
 * \param step_factor Normalized descent step scale epsilon (actual step is step_factor / K).
 * \return DistanceResult containing s_star, distance, and closest_pose.
 */
DistanceResult compute_distance(const Eigen::Matrix4d& h_target,
                                const Se3PolynomialData& poly_data,
                                bool use_cardano = true,
                                double step_factor = 1e-2);

/**
 * \brief Computes the raw first-order analytical distance without local descent refinement.
 * \param h_target Query pose H in SE(3).
 * \param poly_data Precomputed G-polynomial curve representation.
 * \param use_cardano If true, uses closed-form Cardano's formula for cubic roots.
 * \return DistanceResult containing the unrefined s_star, distance, and closest_pose.
 */
DistanceResult compute_distance_raw(const Eigen::Matrix4d& h_target,
                                    const Se3PolynomialData& poly_data,
                                    bool use_cardano = true);

}  // namespace se3_efficient_distance