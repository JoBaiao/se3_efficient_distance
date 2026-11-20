#pragma once

#include "se3_efficient_distance/analytical_distance.hpp"
#include "se3_efficient_distance/polynomial_curves.hpp"

#include <Eigen/Dense>
#include <functional>
#include <string>

namespace se3_efficient_distance::example {

enum class CurveType {
    Harmonic,
    Lemniscate,
    Square,
    Trefoil
};

std::string to_string(CurveType type);

std::function<Eigen::Matrix4d(double)> get_analytical_curve(CurveType type);

void export_and_plot(const std::string& curve_name,
                     const Se3PolynomialData& poly_data,
                     const Eigen::Matrix4d& h_query,
                     const DistanceResult& result);

}  // namespace se3_efficient_distance::example