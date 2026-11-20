#include "example_utils.hpp"
#include "se3_efficient_distance/analytical_distance.hpp"
#include "se3_efficient_distance/polynomial_curves.hpp"
#include "se3_efficient_distance/se3_basic_math.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace se3_efficient_distance;
using example::CurveType;

int main() {
    // Select curve topology: Harmonic, Lemniscate, Square, or Trefoil
    const CurveType selected_curve = CurveType::Trefoil;
    const int k_segs = 11;

    auto analytical_curve = example::get_analytical_curve(selected_curve);

    // Sample K+1 poses in SE(3) along the closed reference curve (C_K = C_0)
    std::vector<Eigen::Matrix4d> sampled_poses;
    sampled_poses.reserve(k_segs + 1);
    for (int k = 0; k <= k_segs; ++k) {
        double s = static_cast<double>(k) / k_segs;
        sampled_poses.push_back(analytical_curve(s));
    }

    // Construct the C^1 quadratic G-polynomial reference curve in O(K)
    Se3PolynomialData reference_curve = create_se3_pol(sampled_poses);

    // Define a generic query pose H in SE(3)
    Eigen::Matrix4d h_query = Eigen::Matrix4d::Identity();
    h_query.block<3, 1>(0, 3) = Eigen::Vector3d(1.35, 0.35, 0.25);            // Translation (x, y, z)
    h_query.block<3, 3>(0, 0) = exp_so3(Eigen::Vector3d(0.35, -0.25, 0.60));  // Rotation in SO(3)

    // Compute the distance and closest point on the reference curve
    DistanceResult result = compute_distance(h_query, reference_curve);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Curve topology       : " << example::to_string(selected_curve) << "\n";
    std::cout << "Optimal parameter s* : " << result.s_star << "\n";
    std::cout << "SE(3) distance D(H)  : " << result.distance << "\n";
    std::cout << "Closest pose C(s*)   :\n" << result.closest_pose << "\n\n";

    example::export_and_plot(example::to_string(selected_curve), reference_curve, h_query, result);

    return 0;
}