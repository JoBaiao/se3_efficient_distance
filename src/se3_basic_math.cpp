#include "se3_efficient_distance/se3_basic_math.hpp"

#include <algorithm>
#include <cmath>

namespace se3_efficient_distance {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d w_hat;
    w_hat << 0.0, -w.z(), w.y(),
             w.z(), 0.0, -w.x(),
             -w.y(), w.x(), 0.0;
    return w_hat;
}

Eigen::Matrix4d inv_se3(const Eigen::Matrix4d& h) {
    Eigen::Matrix4d h_inv = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d r_t = h.block<3, 3>(0, 0).transpose();
    h_inv.block<3, 3>(0, 0) = r_t;
    h_inv.block<3, 1>(0, 3) = -r_t * h.block<3, 1>(0, 3);
    return h_inv;
}

Eigen::Vector3d vee_so3(const Eigen::Matrix3d& w_mat) {
    return Eigen::Vector3d(w_mat(2, 1), w_mat(0, 2), w_mat(1, 0));
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& phi) {
    double th = phi.norm();
    Eigen::Matrix3d w_mat = skew(phi);

    if (th < 1e-8) {
        return Eigen::Matrix3d::Identity() + w_mat + 0.5 * (w_mat * w_mat);
    }

    double a_val = std::sin(th) / th;
    double b_val = (1.0 - std::cos(th)) / (th * th);
    return Eigen::Matrix3d::Identity() + a_val * w_mat + b_val * (w_mat * w_mat);
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d& r) {
    double c = (r.trace() - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    double th = std::acos(c);

    if (th < 1e-8) {
        Eigen::Matrix3d w_mat = 0.5 * (r - r.transpose());
        return vee_so3(w_mat);
    }

    if (kPi - th < 1e-6) {
        Eigen::Matrix3d a_mat = (r + Eigen::Matrix3d::Identity()) * 0.5;
        Eigen::Vector3d axis(
            std::sqrt(std::max(a_mat(0, 0), 0.0)),
            std::sqrt(std::max(a_mat(1, 1), 0.0)),
            std::sqrt(std::max(a_mat(2, 2), 0.0))
        );

        if (r(2, 1) - r(1, 2) < 0.0) axis.x() = -axis.x();
        if (r(0, 2) - r(2, 0) < 0.0) axis.y() = -axis.y();
        if (r(1, 0) - r(0, 1) < 0.0) axis.z() = -axis.z();

        double n = axis.norm();
        if (n < 1e-12) {
            Eigen::Matrix3d w_mat = (r - r.transpose()) / (2.0 * std::sin(th));
            axis = vee_so3(w_mat);
            n = axis.norm();
            axis = (n < 1e-12) ? Eigen::Vector3d(1.0, 0.0, 0.0) : (axis / n);
        } else {
            axis /= n;
        }
        return axis * th;
    }

    Eigen::Matrix3d w_mat = (r - r.transpose()) / (2.0 * std::sin(th));
    return vee_so3(w_mat) * th;
}

Eigen::Matrix3d jac_left_so3(const Eigen::Vector3d& phi) {
    double th = phi.norm();
    Eigen::Matrix3d w_mat = skew(phi);

    if (th < 1e-8) {
        return Eigen::Matrix3d::Identity() + 0.5 * w_mat + (1.0 / 6.0) * (w_mat * w_mat);
    }

    double th2 = th * th;
    double a_val = (1.0 - std::cos(th)) / th2;
    double b_val = (th - std::sin(th)) / (th2 * th);
    return Eigen::Matrix3d::Identity() + a_val * w_mat + b_val * (w_mat * w_mat);
}

Eigen::Matrix3d inv_jac_left_so3(const Eigen::Vector3d& phi) {
    double th = phi.norm();
    Eigen::Matrix3d w_mat = skew(phi);

    if (th < 1e-8) {
        return Eigen::Matrix3d::Identity() - 0.5 * w_mat + (1.0 / 12.0) * (w_mat * w_mat);
    }

    double half = 0.5 * th;
    double cot_half = std::cos(half) / std::sin(half);
    double th2 = th * th;
    return Eigen::Matrix3d::Identity() - 0.5 * w_mat +
           (1.0 / th2) * (1.0 - th * 0.5 * cot_half) * (w_mat * w_mat);
}

Eigen::Matrix4d exp_se3(const Eigen::Matrix4d& a) {
    Eigen::Matrix3d w_hat = a.block<3, 3>(0, 0);
    Eigen::Vector3d v = a.block<3, 1>(0, 3);
    Eigen::Vector3d phi = vee_so3(w_hat);

    Eigen::Matrix4d h = Eigen::Matrix4d::Identity();
    h.block<3, 3>(0, 0) = exp_so3(phi);
    h.block<3, 1>(0, 3) = jac_left_so3(phi) * v;
    return h;
}

Eigen::Matrix4d log_se3(const Eigen::Matrix4d& h) {
    Eigen::Matrix3d r = h.block<3, 3>(0, 0);
    Eigen::Vector3d p = h.block<3, 1>(0, 3);

    Eigen::Vector3d phi = log_so3(r);
    Eigen::Vector3d v = inv_jac_left_so3(phi) * p;

    Eigen::Matrix4d a = Eigen::Matrix4d::Zero();
    a.block<3, 3>(0, 0) = skew(phi);
    a.block<3, 1>(0, 3) = v;
    return a;
}

Eigen::Matrix<double, 6, 1> vec(const Eigen::Matrix4d& a) {
    Eigen::Matrix<double, 6, 1> xi;
    xi.head<3>() = a.block<3, 1>(0, 3);
    xi.tail<3>() = vee_so3(a.block<3, 3>(0, 0));
    return xi;
}

Eigen::Matrix4d vec_inv(const Eigen::Matrix<double, 6, 1>& a) {
    Eigen::Matrix4d a_mat = Eigen::Matrix4d::Zero();
    a_mat.block<3, 3>(0, 0) = skew(a.tail<3>());
    a_mat.block<3, 1>(0, 3) = a.head<3>();
    return a_mat;
}

Eigen::Matrix<double, 6, 6> left_jac(const Eigen::Matrix4d& a) {
    double phi = std::sqrt(a(0, 1) * a(0, 1) + a(0, 2) * a(0, 2) + a(1, 2) * a(1, 2));

    Eigen::Matrix3d vs = skew(a.block<3, 1>(0, 3));
    Eigen::Matrix3d ws = a.block<3, 3>(0, 0);

    Eigen::Matrix<double, 6, 6> a_hat;
    a_hat.block<3, 3>(0, 0) = ws;
    a_hat.block<3, 3>(0, 3) = vs;
    a_hat.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    a_hat.block<3, 3>(3, 3) = ws;

    Eigen::Matrix<double, 6, 6> a0 = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 6> a1 = a_hat;
    Eigen::Matrix<double, 6, 6> a2 = a1 * a_hat;
    Eigen::Matrix<double, 6, 6> a3 = a2 * a_hat;
    Eigen::Matrix<double, 6, 6> a4 = a3 * a_hat;

    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;

    if (phi < 1e-6) {
        double phi2 = phi * phi;
        c1 = 0.5;
        c2 = 1.0 / 6.0;
        c3 = (1.0 / 24.0) - (phi2 / 360.0);
        c4 = (1.0 / 120.0) - (phi2 / 2520.0);
    } else {
        double phi2 = phi * phi;
        double phi3 = phi2 * phi;
        double phi4 = phi3 * phi;
        double phi5 = phi4 * phi;

        double cphi = std::cos(phi);
        double sphi = std::sin(phi);
        double phicphi = phi * cphi;
        double phisphi = phi * sphi;

        c1 = (4.0 - phisphi - 4.0 * cphi) / (2.0 * phi2);
        c2 = (4.0 * phi - 5.0 * sphi + phicphi) / (2.0 * phi3);
        c3 = (2.0 - phisphi - 2.0 * cphi) / (2.0 * phi4);
        c4 = (2.0 * phi - 3.0 * sphi + phicphi) / (2.0 * phi5);
    }

    return a0 + c1 * a1 + c2 * a2 + c3 * a3 + c4 * a4;
}

Eigen::Matrix<double, 6, 6> i_se3(const Eigen::Matrix4d& a) {
    Eigen::Matrix<double, 6, 6> j_mat = left_jac(a);
    Eigen::Matrix4d h = exp_se3(-a);

    Eigen::Matrix3d q = h.block<3, 3>(0, 0);
    Eigen::Vector3d d = h.block<3, 1>(0, 3);

    Eigen::Vector3d a0 = q * j_mat.block<3, 1>(0, 0);
    Eigen::Vector3d a1 = q * j_mat.block<3, 1>(0, 1);
    Eigen::Vector3d a2 = q * j_mat.block<3, 1>(0, 2);

    Eigen::Vector3d a3 = -skew(q * j_mat.block<3, 1>(3, 3)) * d + q * j_mat.block<3, 1>(0, 3);
    Eigen::Vector3d a4 = -skew(q * j_mat.block<3, 1>(3, 4)) * d + q * j_mat.block<3, 1>(0, 4);
    Eigen::Vector3d a5 = -skew(q * j_mat.block<3, 1>(3, 5)) * d + q * j_mat.block<3, 1>(0, 5);

    Eigen::Matrix<double, 6, 6> i_mat;
    i_mat.block<3, 1>(0, 0) = a0;
    i_mat.block<3, 1>(0, 1) = a1;
    i_mat.block<3, 1>(0, 2) = a2;
    i_mat.block<3, 1>(0, 3) = a3;
    i_mat.block<3, 1>(0, 4) = a4;
    i_mat.block<3, 1>(0, 5) = a5;

    i_mat.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    i_mat.block<3, 1>(3, 3) = a0;
    i_mat.block<3, 1>(3, 4) = a1;
    i_mat.block<3, 1>(3, 5) = a2;

    return i_mat;
}

Eigen::Matrix4d l_se3(const Eigen::Matrix4d& a,
                      const Eigen::Matrix4d& e,
                      Eigen::Matrix3d* i1_inv,
                      Eigen::Matrix3d* b_inv) {
    Eigen::Matrix3d loc_i1_inv;
    Eigen::Matrix3d loc_b_inv;

    if (i1_inv == nullptr || b_inv == nullptr) {
        Eigen::Matrix<double, 6, 6> i_mat = i_se3(-a);
        loc_i1_inv = i_mat.block<3, 3>(0, 0).inverse();
        Eigen::Matrix3d i2 = i_mat.block<3, 3>(0, 3);
        loc_b_inv = loc_i1_inv * i2 * loc_i1_inv;
    } else {
        loc_i1_inv = *i1_inv;
        loc_b_inv = *b_inv;
    }

    Eigen::Matrix<double, 6, 1> xi = vec(e);
    Eigen::Matrix<double, 6, 1> res;
    res.head<3>() = loc_i1_inv * xi.head<3>() - loc_b_inv * xi.tail<3>();
    res.tail<3>() = loc_i1_inv * xi.tail<3>();

    if (i1_inv != nullptr) *i1_inv = loc_i1_inv;
    if (b_inv != nullptr) *b_inv = loc_b_inv;

    return vec_inv(res);
}

}  // namespace se3_efficient_distance