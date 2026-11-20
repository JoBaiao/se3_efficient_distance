#include "example_utils.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace se3_efficient_distance::example {

namespace {

Eigen::Matrix4d translation_matrix(const Eigen::Vector3d& t) {
    Eigen::Matrix4d h_mat = Eigen::Matrix4d::Identity();
    h_mat.block<3, 1>(0, 3) = t;
    return h_mat;
}

Eigen::Matrix4d rotx_matrix(double angle) {
    Eigen::Matrix4d h_mat = Eigen::Matrix4d::Identity();
    double c = std::cos(angle);
    double s = std::sin(angle);
    h_mat(1, 1) = c; h_mat(1, 2) = -s;
    h_mat(2, 1) = s; h_mat(2, 2) = c;
    return h_mat;
}

Eigen::Matrix4d roty_matrix(double angle) {
    Eigen::Matrix4d h_mat = Eigen::Matrix4d::Identity();
    double c = std::cos(angle);
    double s = std::sin(angle);
    h_mat(0, 0) = c; h_mat(0, 2) = s;
    h_mat(2, 0) = -s; h_mat(2, 2) = c;
    return h_mat;
}

Eigen::Matrix4d rotz_matrix(double angle) {
    Eigen::Matrix4d h_mat = Eigen::Matrix4d::Identity();
    double c = std::cos(angle);
    double s = std::sin(angle);
    h_mat(0, 0) = c; h_mat(0, 1) = -s;
    h_mat(1, 0) = s; h_mat(1, 1) = c;
    return h_mat;
}

void write_matrix_json(std::ostream& os, const Eigen::Matrix4d& mat) {
    os << "[";
    for (int r = 0; r < 4; ++r) {
        os << "[";
        for (int c = 0; c < 4; ++c) {
            os << mat(r, c) << (c < 3 ? ", " : "");
        }
        os << "]" << (r < 3 ? ", " : "");
    }
    os << "]";
}

}  // namespace

std::string to_string(CurveType type) {
    switch (type) {
        case CurveType::Harmonic:   return "Harmonic";
        case CurveType::Lemniscate: return "Lemniscate";
        case CurveType::Square:     return "Square";
        case CurveType::Trefoil:    return "Trefoil";
    }
    return "Unknown";
}

std::function<Eigen::Matrix4d(double)> get_analytical_curve(CurveType type) {
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

    switch (type) {
        case CurveType::Harmonic: {
            constexpr double a_x = 0.3;
            constexpr double a_y = 0.2;
            constexpr double a_z = 0.2;
            constexpr double z_freq = 3.0;
            return [=](double s) -> Eigen::Matrix4d {
                double t = kTwoPi * s;
                double x = std::cos(t) + a_x * std::sin(3.0 * t);
                double y = std::sin(t) + a_y * std::cos(4.0 * t);
                double z = a_z * std::sin(z_freq * t);
                return translation_matrix(Eigen::Vector3d(x, y, z)) *
                       rotx_matrix(t) * roty_matrix(2.0 * t + 0.5);
            };
        }
        case CurveType::Lemniscate: {
            constexpr double a_val = 1.0;
            constexpr double z_amp = 0.2;
            return [=](double s) -> Eigen::Matrix4d {
                double t = kTwoPi * s;
                double den = 1.0 + std::sin(t) * std::sin(t);
                double x = a_val * std::cos(t) / den;
                double y = a_val * std::sin(t) * std::cos(t) / den;
                double z = z_amp * std::sin(t);
                return translation_matrix(Eigen::Vector3d(x, y, z)) *
                       rotx_matrix(t) * roty_matrix(t);
            };
        }
        case CurveType::Square: {
            constexpr double a_val = 1.0;
            constexpr double z_amp = 0.0;
            return [=](double s) -> Eigen::Matrix4d {
                double t = kTwoPi * s;
                double den = std::sqrt(1.0 - 0.5 * std::sin(2.0 * t) * std::sin(2.0 * t));
                double x = a_val * std::cos(t) / den;
                double y = a_val * std::sin(t) / den;
                double z = z_amp * std::sin(4.0 * t);
                return translation_matrix(Eigen::Vector3d(x, y, z)) * rotz_matrix(t);
            };
        }
        case CurveType::Trefoil: {
            constexpr double sx = 0.3;
            constexpr double sy = 0.3;
            constexpr double sz = 0.3;
            constexpr double shear = 0.0;
            return [=](double s) -> Eigen::Matrix4d {
                double t = kTwoPi * s;
                double base_x = std::sin(t) + 2.0 * std::sin(2.0 * t);
                double base_y = std::cos(t) - 2.0 * std::cos(2.0 * t);
                double base_z = -std::sin(3.0 * t);
                double x = sx * base_x + shear * base_y;
                double y = sy * base_y + shear * base_x;
                double z = sz * base_z;
                return translation_matrix(Eigen::Vector3d(x, y, z)) * rotz_matrix(t);
            };
        }
    }
    throw std::invalid_argument("Unsupported CurveType.");
}

void export_and_plot(const std::string& curve_name,
                     const Se3PolynomialData& poly_data,
                     const Eigen::Matrix4d& h_query,
                     const DistanceResult& result) {
    namespace fs = std::filesystem;

#ifdef EXAMPLES_DIR
    fs::path examples_dir(EXAMPLES_DIR);
#else
    fs::path examples_dir = fs::exists("examples") ? fs::path("examples") : fs::path("../examples");
#endif

    fs::path json_path = fs::temp_directory_path() / "se3_distance_example_data.json";
    fs::path script_path = examples_dir / "plot_example.py";
    fs::path output_img = examples_dir / "se3_distance_example.png";

    std::ofstream out(json_path);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not write temporary visualization data to " << json_path << "\n";
        return;
    }

    constexpr int kNumTrajSamples = 250;
    constexpr int kNumFrameSamples = 20;

    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"curve_name\": \"" << curve_name << "\",\n";
    out << "  \"s_star\": " << result.s_star << ",\n";
    out << "  \"distance\": " << result.distance << ",\n";

    out << "  \"h_query\": ";
    write_matrix_json(out, h_query);
    out << ",\n";

    out << "  \"h_closest\": ";
    write_matrix_json(out, result.closest_pose);
    out << ",\n";

    out << "  \"trajectory_points\": [\n";
    for (int i = 0; i <= kNumTrajSamples; ++i) {
        double s = static_cast<double>(i) / kNumTrajSamples;
        Eigen::Vector3d p = poly_data.curve(s).block<3, 1>(0, 3);
        out << "    [" << p.x() << ", " << p.y() << ", " << p.z() << "]"
            << (i < kNumTrajSamples ? ",\n" : "\n");
    }
    out << "  ],\n";

    out << "  \"curve_frames\": [\n";
    for (int i = 0; i < kNumFrameSamples; ++i) {
        double s = static_cast<double>(i) / kNumFrameSamples;
        out << "    ";
        write_matrix_json(out, poly_data.curve(s));
        out << (i + 1 < kNumFrameSamples ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::string cmd = "python3 \"" + script_path.string() + "\" \"" +
                      json_path.string() + "\" \"" + output_img.string() + "\"";
    int ret = std::system(cmd.c_str());
    fs::remove(json_path);

    if (ret == 0) {
        std::cout << "Saved 3D visualization to: " << output_img << "\n";
    } else {
        std::cerr << "Warning: Visualization script failed (ensure python3 and matplotlib are installed).\n";
    }
}

}  // namespace se3_efficient_distance::example