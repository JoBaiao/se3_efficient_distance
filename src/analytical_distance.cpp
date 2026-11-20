#include "se3_efficient_distance/analytical_distance.hpp"
#include "se3_efficient_distance/se3_basic_math.hpp"

#include <cmath>
#include <limits>

namespace se3_efficient_distance {

DistanceResult compute_distance_raw(const Eigen::Matrix4d& h_target,
                                    const Se3PolynomialData& poly_data,
                                    bool use_cardano) {
    int k_segs = poly_data.k_segs;
    double s_best = 0.0;
    double f_best = std::numeric_limits<double>::max();

    for (int i = 0; i < k_segs; ++i) {
        auto [s_local, val] = min_seg_approx(
            h_target, poly_data.c_mats[i], poly_data.e1_mats[i], poly_data.e2_mats[i], use_cardano);
        if (val < f_best) {
            f_best = val;
            s_best = (i + s_local) / k_segs;
        }
    }

    s_best = std::fmod(s_best, 1.0);
    if (s_best < 0.0) s_best += 1.0;

    Eigen::Matrix4d h_closest = poly_data.curve(s_best);
    double dist = log_se3(inv_se3(h_target) * h_closest).norm();
    return {s_best, dist, h_closest};
}

DistanceResult compute_distance(const Eigen::Matrix4d& h_target,
                                const Se3PolynomialData& poly_data,
                                bool use_cardano,
                                double step_factor) {
    int k_segs = poly_data.k_segs;
    Eigen::Matrix4d h_target_inv = inv_se3(h_target);

    auto dist_fun = [&](double s) -> double {
        return log_se3(h_target_inv * poly_data.curve(s)).norm();
    };

    // 1. Polynomial initial guess across all segments
    double s_best = 0.0;
    double f_best = std::numeric_limits<double>::max();
    for (int i = 0; i < k_segs; ++i) {
        auto [s_local, val] = min_seg_approx(
            h_target, poly_data.c_mats[i], poly_data.e1_mats[i], poly_data.e2_mats[i], use_cardano);
        if (val < f_best) {
            f_best = val;
            s_best = (i + s_local) / k_segs;
        }
    }

    // 2. Local descent refinement using central difference direction
    double h_diff = 1e-3 / k_segs;
    double dfun_h = dist_fun(s_best + h_diff) - dist_fun(s_best - h_diff);
    double ds_base = step_factor / k_segs;
    double ds = (dfun_h > 0.0) ? -ds_base : ds_base;

    double f_cur = dist_fun(s_best);
    while (true) {
        double f_new = dist_fun(s_best + ds);
        if (f_new < f_cur) {
            s_best += ds;
            f_cur = f_new;
        } else {
            break;
        }
    }

    s_best = std::fmod(s_best, 1.0);
    if (s_best < 0.0) s_best += 1.0;

    Eigen::Matrix4d h_closest = poly_data.curve(s_best);
    return {s_best, f_cur, h_closest};
}

}  // namespace se3_efficient_distance