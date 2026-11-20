#include "se3_efficient_distance/polynomial_curves.hpp"
#include "se3_efficient_distance/se3_basic_math.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/Polynomials>

namespace se3_efficient_distance {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

std::vector<double> solve_cubic(double a, double b, double c, double d) {
    std::vector<double> real_roots;
    if (std::abs(a) < 1e-12) {
        if (std::abs(b) < 1e-12) return {};
        double disc = c * c - 4.0 * b * d;
        if (disc < 0.0) return {};
        double sqrt_disc = std::sqrt(disc);
        real_roots.push_back((-c + sqrt_disc) / (2.0 * b));
        real_roots.push_back((-c - sqrt_disc) / (2.0 * b));
        return real_roots;
    }

    double p = (3.0 * a * c - b * b) / (3.0 * a * a);
    double q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    double delta = (q * q / 4.0) + (p * p * p / 27.0);

    if (delta > 0.0) {
        double sqrt_delta = std::sqrt(delta);
        double u = std::cbrt(-q / 2.0 + sqrt_delta);
        double v = std::cbrt(-q / 2.0 - sqrt_delta);
        real_roots.push_back(u + v - b / (3.0 * a));
    } else if (std::abs(delta) < 1e-12) {
        double u = std::cbrt(-q / 2.0);
        real_roots.push_back(2.0 * u - b / (3.0 * a));
        real_roots.push_back(-u - b / (3.0 * a));
    } else {
        double r = std::sqrt(-p * p * p / 27.0);
        double arg = std::max(-1.0, std::min(1.0, -q / (2.0 * r)));
        double phi = std::acos(arg);
        double t = 2.0 * std::cbrt(r);
        real_roots.push_back(t * std::cos(phi / 3.0) - b / (3.0 * a));
        real_roots.push_back(t * std::cos((phi + 2.0 * kPi) / 3.0) - b / (3.0 * a));
        real_roots.push_back(t * std::cos((phi + 4.0 * kPi) / 3.0) - b / (3.0 * a));
    }
    return real_roots;
}

std::pair<double, double> min_seg_approx(const Eigen::Matrix4d& h_target,
                                         const Eigen::Matrix4d& c_mat,
                                         const Eigen::Matrix4d& e1_mat,
                                         const Eigen::Matrix4d& e2_mat,
                                         bool use_cardano) {
    Eigen::Matrix4d a_mat = log_se3(inv_se3(c_mat) * h_target);

    Eigen::Matrix<double, 6, 6> i_mat = i_se3(-a_mat);
    Eigen::Matrix3d i1_inv = i_mat.block<3, 3>(0, 0).inverse();
    Eigen::Matrix3d b_inv = i1_inv * i_mat.block<3, 3>(0, 3) * i1_inv;

    Eigen::Matrix4d u1 = l_se3(a_mat, e1_mat, &i1_inv, &b_inv);
    Eigen::Matrix4d u2 = l_se3(a_mat, e2_mat, &i1_inv, &b_inv);

    auto inner_prod = [](const Eigen::Matrix4d& m1, const Eigen::Matrix4d& m2) -> double {
        return static_cast<double>(
            (m1.transpose().cast<long double>() * m2.cast<long double>()).trace());
    };

    double b0 = inner_prod(a_mat, a_mat);
    double b1 = -2.0 * inner_prod(a_mat, u1);
    double b2 = -2.0 * inner_prod(a_mat, u2) + inner_prod(u1, u1);
    double b3 = 2.0 * inner_prod(u1, u2);
    double b4 = inner_prod(u2, u2);

    auto eval_pol = [b0, b1, b2, b3, b4](double s) {
        return (((b4 * s + b3) * s + b2) * s + b1) * s + b0;
    };

    std::vector<std::pair<double, double>> candidates = {
        {0.0, eval_pol(0.0)},
        {1.0, eval_pol(1.0)}
    };

    if (use_cardano) {
        std::vector<double> roots = solve_cubic(4.0 * b4, 3.0 * b3, 2.0 * b2, b1);
        for (double r : roots) {
            if (r > 0.0 && r < 1.0) {
                candidates.push_back({r, eval_pol(r)});
            }
        }
    } else {
        Eigen::VectorXd coeff_vec(4);
        coeff_vec << b1, 2.0 * b2, 3.0 * b3, 4.0 * b4;

        Eigen::PolynomialSolver<double, 3> solver;
        solver.compute(coeff_vec);

        const auto& roots = solver.roots();
        for (int i = 0; i < roots.size(); ++i) {
            if (std::abs(roots[i].imag()) <= 1e-12) {
                double r = roots[i].real();
                if (r > 0.0 && r < 1.0) {
                    candidates.push_back({r, eval_pol(r)});
                }
            }
        }
    }

    auto min_it = std::min_element(
        candidates.begin(), candidates.end(),
        [](const auto& p1, const auto& p2) { return p1.second < p2.second; });
    return *min_it;
}

Se3PolynomialData create_se3_pol(const std::vector<Eigen::Matrix4d>& sample_points) {
    if (sample_points.size() < 3) {
        throw std::invalid_argument("sample_points must contain at least 3 poses (K >= 2 segments).");
    }

    int k_segs = static_cast<int>(sample_points.size()) - 1;
    const std::vector<Eigen::Matrix4d>& c_mats = sample_points;

    std::vector<Eigen::Matrix4d> a_mats(k_segs);
    std::vector<Eigen::Matrix<double, 6, 6>> i_mats(k_segs);
    std::vector<Eigen::Matrix<double, 6, 1>> a_vecs(k_segs);

    for (int k = 0; k < k_segs; ++k) {
        a_mats[k] = log_se3(inv_se3(c_mats[k]) * c_mats[k + 1]);
        i_mats[k] = i_se3(a_mats[k]);
        a_vecs[k] = vec(a_mats[k]);
    }

    Eigen::Matrix<double, 6, 6> m_k = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 1> v_k = Eigen::Matrix<double, 6, 1>::Zero();

    for (int k = 0; k < k_segs; ++k) {
        v_k = 2.0 * a_vecs[k] - i_mats[k] * v_k;
        m_k = -i_mats[k] * m_k;
    }

    Eigen::Matrix<double, 6, 6> left_side = Eigen::Matrix<double, 6, 6>::Identity() - m_k;
    Eigen::Matrix<double, 6, 1> e_0 = left_side.colPivHouseholderQr().solve(v_k);

    std::vector<Eigen::Matrix<double, 6, 1>> e_vecs(k_segs);
    e_vecs[0] = e_0;

    for (int k = 0; k < k_segs - 1; ++k) {
        e_vecs[k + 1] = 2.0 * a_vecs[k] - i_mats[k] * e_vecs[k];
    }

    std::vector<Eigen::Matrix4d> e1_mats(k_segs);
    std::vector<Eigen::Matrix4d> e2_mats(k_segs);
    for (int k = 0; k < k_segs; ++k) {
        e1_mats[k] = vec_inv(e_vecs[k]);
        e2_mats[k] = a_mats[k] - e1_mats[k];
    }

    auto curve_func = [c_mats, e1_mats, e2_mats, k_segs](double s) -> Eigen::Matrix4d {
        double s_mod = std::fmod(s, 1.0);
        if (s_mod < 0.0) s_mod += 1.0;

        int ind = static_cast<int>(std::floor(s_mod * k_segs));
        if (ind >= k_segs) ind = k_segs - 1;

        double snorm = s_mod * k_segs - ind;
        return c_mats[ind] * exp_se3(e1_mats[ind] * snorm + e2_mats[ind] * (snorm * snorm));
    };

    return {curve_func, c_mats, e1_mats, e2_mats, k_segs};
}

Se3PolynomialData create_se3_pol(const std::function<Eigen::Matrix4d(double)>& target_curve, int k_segs) {
    if (k_segs < 2) {
        throw std::invalid_argument("k_segs must be >= 2.");
    }

    std::vector<Eigen::Matrix4d> sample_points(k_segs + 1);
    for (int k = 0; k <= k_segs; ++k) {
        sample_points[k] = target_curve(static_cast<double>(k) / k_segs);
    }
    return create_se3_pol(sample_points);
}

}  // namespace se3_efficient_distance