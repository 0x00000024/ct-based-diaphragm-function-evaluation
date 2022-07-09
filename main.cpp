#define CGAL_USE_BASIC_VIEWER

#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/draw_surface_mesh.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/remove_outliers.h>

#include <cxxabi.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

#include "ThreadPool.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_3;
typedef Kernel_3::Point_3 Point_3;
typedef Kernel_3::Vector_3 Vector_3;
typedef CGAL::Point_set_3<Point_3, Vector_3> Point_set_3;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef CGAL::Surface_mesh<Point> Mesh;

void get_one_side_mask(const Mesh &lung_base_mask_mesh, const std::string &direction, int x_min, int x_max, int y_min,
                       int y_max, const std::filesystem::path &result_dir_path) {
    unsigned int n = std::thread::hardware_concurrency();
    ThreadPool pool(n - 20);
    std::cout << n - 20 << " concurrent threads are generated.\n";

    std::vector<std::future<Point>> results;
    int step = (direction == "l2r") ? 1 : -1;
    if (direction != "l2r") {
        std::swap(y_min, y_max);
    }
    std::cout << "step" << step << std::endl;
    std::cout << "y_min" << y_min << std::endl;
    std::cout << "y_max" << y_max << std::endl;

    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = x_max; x >= x_min; x -= std::abs(step)) {
        results.emplace_back(
                pool.enqueue([x, x_min, x_max, y_min, y_max, step, lung_base_mask_mesh] {
                    for (int y = y_min; y != y_max + step; y += step) {
                        int z = 0;
                        std::vector<Point> polyline;
                        polyline.push_back(Point(x, y - step, z));
                        Point curr_point = Point(x, y, z);
                        polyline.push_back(curr_point);
                        if (CGAL::Polygon_mesh_processing::do_intersect(lung_base_mask_mesh, polyline)) {
                            std::cout << "[" << double((x_max - x)) / double((x_max - x_min)) * 100 << "%" << "] ";
                            std::cout << x_max - x << " planes have been analyzed, " << x - x_min
                                      << " planes are left." << std::endl;
                            std::cout << "Found contact point " << curr_point << std::endl;
                            return curr_point;
                        }
                    }
                })
        );
        std::cout << "[" << double((x_max - x)) / double((x_max - x_min)) * 100 << "%" << "] ";
        std::cout << x_max - x << " planes have been enqueued, " << x - x_min << " planes are left." << std::endl;
    }

    for (auto &&result: results) {
        std::cout << "Waiting..." << std::endl;
        result.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "The total time spent is: " << duration.count() << "(s).." << std::endl;

    Point_set point_set;
    for (auto &&result: results) {
        Point tmp = result.get();
        if (tmp[0] == 0.0) {
            continue;
        }
        point_set.insert(tmp);
    }

    std::cout << "The number of points generated: " << point_set.size() << std::endl;

    std::filesystem::path lung_base_mask_3d_point_cloud_filename("5.lung_base_mask_" + direction + "_3d_point_cloud.ply");
    std::filesystem::path lung_base_mask_3d_point_cloud_file_path = result_dir_path / lung_base_mask_3d_point_cloud_filename;
    CGAL::IO::write_PLY(lung_base_mask_3d_point_cloud_file_path, point_set);
    std::cout << "The generated file was saved to " << lung_base_mask_3d_point_cloud_file_path << std::endl;
}

bool x_min_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[0] > b[0]);
}

bool x_max_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[0] < b[0]);
}

bool y_min_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[1] > b[1]);
}

bool y_max_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[1] < b[1]);
}

bool z_min_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[2] > b[2]);
}

bool z_max_comp(CGAL::Point_3<CGAL::Epick> a, CGAL::Point_3<CGAL::Epick> b) {
    return (a[2] < b[2]);
}

void outlier_removal_lung_diaphragmatic_interface(
        const std::filesystem::path &point_cloud_file_path,
        const std::filesystem::path &point_cloud_outlier_removed_file_path) {
    Point_set_3 points;
    std::ifstream stream(point_cloud_file_path, std::ios_base::binary);
    if (!stream) {
        std::cerr << "Error: cannot read file " << point_cloud_file_path
                  << std::endl;
        return;
    }
    stream >> points;

    std::cout << "Read " << points.size() << " point(s)" << std::endl;
    if (points.empty()) {
        std::cerr << "Error: empty point cloud file" << point_cloud_file_path
                  << std::endl;
        return;
    }

    auto rout_it = CGAL::remove_outliers<CGAL::Sequential_tag>(
            points, 24, points.parameters().threshold_percent(5.0));
    points.remove(rout_it, points.end());
    std::cout << points.number_of_removed_points() << " point(s) are outliers."
              << std::endl;
    points.collect_garbage();

    std::cout << "The number of points generated: " << points.size() << std::endl;
    CGAL::IO::write_PLY(point_cloud_outlier_removed_file_path, points);
    std::cout << "The generated file was saved to "
              << point_cloud_outlier_removed_file_path << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " /root/diaphragm-evaluation/result/10003382/ex "
                     "lung_base_3d_point_cloud.ply"
                  << std::endl;
    }

    std::filesystem::path result_dir_path(argv[1]);
    std::filesystem::path point_cloud_filename(argv[2]);
    std::filesystem::path point_cloud_file_path =
            result_dir_path / point_cloud_filename;
    std::filesystem::path mesh_filename("6.lung_base_mesh.ply");
    std::filesystem::path mesh_file_path = result_dir_path / mesh_filename;
    std::filesystem::path contact_point_cloud_filename(
            "7.lung_diaphragm_contact_surface_3d_point_cloud.ply");
    std::filesystem::path contact_point_cloud_file_path =
            result_dir_path / contact_point_cloud_filename;
    std::filesystem::path contact_point_cloud_outlier_removed_filename(
            "8.lung_diaphragm_contact_surface_3d_point_cloud_outlier_removed(manually).ply");
    std::filesystem::path contact_point_cloud_outlier_removed_file_path =
            result_dir_path / contact_point_cloud_outlier_removed_filename;
    std::filesystem::path lung_base_mask_mesh_filename(
            "4.lung_base_mask_mesh(manually).ply");
    std::filesystem::path lung_base_mask_mesh_file_path =
            result_dir_path / lung_base_mask_mesh_filename;

    Point_set_3 points;
    std::ifstream stream(point_cloud_file_path, std::ios_base::binary);
    if (!stream) {
        std::cerr << "Error: cannot read file " << point_cloud_file_path
                  << std::endl;
        return EXIT_FAILURE;
    }
    stream >> points;

    std::cout << "Read " << points.size() << " point(s)" << std::endl;
    if (points.empty()) return EXIT_FAILURE;

    auto rout_it = CGAL::remove_outliers<CGAL::Sequential_tag>(
            points,
            24,  // Number of neighbors considered for evaluation
            points.parameters().threshold_percent(
                    5.0));  // Percentage of points to remove
    points.remove(rout_it, points.end());
    std::cout << points.number_of_removed_points() << " point(s) are outliers."
              << std::endl;
    // Applying point set processing algorithm to a CGAL::Point_set_3
    // object does not erase the points from memory but place them in
    // the garbage of the object: memory can be freed by the user.
    points.collect_garbage();

    std::cout << "x" << std::endl;
    auto x_min_point = std::max_element(points.points().begin(),
                                        points.points().end(), x_min_comp);
    std::cout << *x_min_point << std::endl;
    int x_min = int((*x_min_point)[0]);
    std::cout << x_min << std::endl;

    auto x_max_point = std::max_element(points.points().begin(),
                                        points.points().end(), x_max_comp);
    std::cout << *x_max_point << std::endl;
    int x_max = std::ceil((*x_max_point)[0]);
    std::cout << x_max << std::endl;

    std::cout << "y" << std::endl;
    auto y_min_point = std::max_element(points.points().begin(),
                                        points.points().end(), y_min_comp);
    std::cout << *y_min_point << std::endl;
    int y_min = int((*y_min_point)[1]);
    std::cout << y_min << std::endl;

    auto y_max_point = std::max_element(points.points().begin(),
                                        points.points().end(), y_max_comp);
    std::cout << *y_max_point << std::endl;
    int y_max = std::ceil((*y_max_point)[1]);
    std::cout << y_max << std::endl;

    std::cout << "z" << std::endl;
    auto z_min_point = std::max_element(points.points().begin(),
                                        points.points().end(), z_min_comp);
    std::cout << *z_min_point << std::endl;
    int z_min = int((*z_min_point)[2]);
    std::cout << z_min << std::endl;

    auto z_max_point = std::max_element(points.points().begin(),
                                        points.points().end(), z_max_comp);
    std::cout << *z_max_point << std::endl;
    int z_max = std::ceil((*z_max_point)[2]);
    std::cout << z_max << std::endl;

    // Triple of indices
    typedef std::array<std::size_t, 3> Facet;
    std::vector<Facet> facets;
    CGAL::advancing_front_surface_reconstruction(points.points().begin(),
                                                 points.points().end(),
                                                 std::back_inserter(facets));
    std::cout << facets.size() << " facet(s) generated by reconstruction."
              << std::endl;

    // Copy points for random access
    std::vector<Point_3> vertices;
    vertices.reserve(points.size());
    std::copy(points.points().begin(), points.points().end(),
              std::back_inserter(vertices));

    CGAL::Surface_mesh<Point_3> lung_base_mesh_3;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vertices, facets,
                                                                lung_base_mesh_3);
    CGAL::IO::write_polygon_mesh(mesh_file_path.string(), lung_base_mesh_3);

    // Intersection
    Mesh lung_base_mesh;
    if (!CGAL::IO::read_polygon_mesh(mesh_file_path.string(), lung_base_mesh)) {
        std::cerr << "Invalid Lung Base Mesh." << std::endl;
        return EXIT_FAILURE;
    }
    unsigned int n = std::thread::hardware_concurrency();

    ThreadPool pool(n - 20);
    std::cout << n - 20 << " concurrent threads are generated.\n";

    std::vector<std::future<Point>> results;
    int step = 10;

    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = x_max; x >= x_min; x -= step) {
        for (int y = y_min; y <= y_max; y += step) {
            results.emplace_back(
                    pool.enqueue([x, y, x_min, x_max, z_min, z_max, lung_base_mesh] {
                        for (int z = z_max; z >= z_min; --z) {
                            std::vector<Point> polyline;
                            polyline.push_back(Point(x, y, z + 1));
                            Point curr_point = Point(x, y, z);
                            polyline.push_back(curr_point);
                            if (CGAL::Polygon_mesh_processing::do_intersect(lung_base_mesh,
                                                                            polyline)) {
                                std::cout << "["
                                          << double((x_max - x)) / double((x_max - x_min)) * 100
                                          << "%"
                                          << "] ";
                                std::cout << x_max - x << " planes have been analyzed, "
                                          << x - x_min << " planes are left." << std::endl;
                                std::cout << "Found contact point " << curr_point << std::endl;
                                return curr_point;
                            }
                        }
                    }));
        }
        std::cout << "[" << double((x_max - x)) / double((x_max - x_min)) * 100
                  << "%"
                  << "] ";
        std::cout << x_max - x << " planes have been enqueued, " << x - x_min
                  << " planes are left." << std::endl;
    }

    for (auto &&result: results) {
        std::cout << "Waiting..." << std::endl;
        result.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "The total time spent is: " << duration.count() << "(s).."
              << std::endl;

    Point_set point_set;
    for (auto &&result: results) {
        Point tmp = result.get();

        if (tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0) {
            continue;
        }
        point_set.insert(tmp);
    }

    std::cout << "The number of points generated: " << point_set.size()
              << std::endl;
    CGAL::IO::write_PLY(contact_point_cloud_file_path, point_set);
    std::cout << "The generated file was saved to "
              << contact_point_cloud_file_path << std::endl;

    outlier_removal_lung_diaphragmatic_interface(
            contact_point_cloud_file_path,
            contact_point_cloud_outlier_removed_file_path);

    Mesh lung_base_mask_mesh;
    if (!CGAL::IO::read_polygon_mesh(lung_base_mask_mesh_file_path.string(), lung_base_mask_mesh)) {
        std::cerr << "Invalid Lung Base Mask Mesh." << std::endl;
        return EXIT_FAILURE;
    }

    get_one_side_mask(lung_base_mask_mesh, "l2r", x_min, x_max, y_min, y_max, result_dir_path);
    get_one_side_mask(lung_base_mask_mesh, "r2l", x_min, x_max, y_min, y_max, result_dir_path);

    return EXIT_SUCCESS;
}
