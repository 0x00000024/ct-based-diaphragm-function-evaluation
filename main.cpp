#define CGAL_USE_BASIC_VIEWER

#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Scale_space_reconstruction_3/Advancing_front_mesher.h>
#include <CGAL/Scale_space_reconstruction_3/Jet_smoother.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/draw_surface_mesh.h>
#include <CGAL/polygon_mesh_processing.h>
#include <cxxabi.h>

#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>

// types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Sphere_3 Sphere_3;
typedef CGAL::Point_set_3<Point_3, Vector_3> Point_set;

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

int main(int argc, char *argv[]) {
    Point_set points;
    std::string fname =
            argc == 1 ? CGAL::data_file_path("lung/ex/max_z.ply") : argv[1];
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [input.xyz/off/ply/las]"
                  << std::endl;
        std::cerr << "Running " << argv[0] << " data/kitten.xyz -1\n";
    }
    std::ifstream stream(fname, std::ios_base::binary);
    if (!stream) {
        std::cerr << "Error: cannot read file " << fname << std::endl;
        return EXIT_FAILURE;
    }
    stream >> points;

    std::cout << "Read " << points.size() << " point(s)" << std::endl;
    if (points.empty()) return EXIT_FAILURE;

    typename Point_set::iterator rout_it =
            CGAL::remove_outliers<CGAL::Sequential_tag>(
                    points,
                    24,  // Number of neighbors considered for evaluation
                    points.parameters().threshold_percent(5.0));  // Percentage of points to remove
    points.remove(rout_it, points.end());
    std::cout << points.number_of_removed_points() << " point(s) are outliers." << std::endl;
    // Applying point set processing algorithm to a CGAL::Point_set_3
    // object does not erase the points from memory but place them in
    // the garbage of the object: memory can be freeed by the user.
    points.collect_garbage();

    std::cout << "x" << std::endl;
    auto x_min_point = std::max_element(points.points().begin(), points.points().end(), x_min_comp);
    std::cout << *x_min_point << std::endl;
    int x_min = int((*x_min_point)[0]);
    std::cout << x_min << std::endl;

    auto x_max_point = std::max_element(points.points().begin(), points.points().end(), x_max_comp);
    std::cout << *x_max_point << std::endl;
    int x_max = std::ceil((*x_max_point)[0]);
    std::cout << x_max << std::endl;

    std::cout << "y" << std::endl;
    auto y_min_point = std::max_element(points.points().begin(), points.points().end(), y_min_comp);
    std::cout << *y_min_point << std::endl;
    int y_min = int((*y_min_point)[1]);
    std::cout << y_min << std::endl;

    auto y_max_point = std::max_element(points.points().begin(), points.points().end(), y_max_comp);
    std::cout << *y_max_point << std::endl;
    int y_max = std::ceil((*y_max_point)[1]);
    std::cout << y_max << std::endl;

    std::cout << "z" << std::endl;
    auto z_min_point = std::max_element(points.points().begin(), points.points().end(), z_min_comp);
    std::cout << *z_min_point << std::endl;
    int z_min = int((*z_min_point)[2]);
    std::cout << z_min << std::endl;

    auto z_max_point = std::max_element(points.points().begin(), points.points().end(), z_max_comp);
    std::cout << *z_max_point << std::endl;
    int z_max = std::ceil((*z_max_point)[2]);
    std::cout << z_max << std::endl;

//    // Compute average spacing using neighborhood of 6 points
//    double spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6);
//    // Simplify using a grid of size 2 * average spacing
//    typename Point_set::iterator gsim_it = CGAL::grid_simplify_point_set(points, 2. * spacing);
//    points.remove(gsim_it, points.end());
//    std::cout << points.number_of_removed_points()
//              << " point(s) removed after simplification." << std::endl;
//    points.collect_garbage();

    CGAL::Scale_space_surface_reconstruction_3<Kernel> reconstruct(points.points().begin(), points.points().end());
    typedef std::array<std::size_t, 3> Facet; // Triple of indices
    std::vector <Facet> facets;
    CGAL::advancing_front_surface_reconstruction(points.points().begin(), points.points().end(), std::back_inserter(facets));
    std::cout << facets.size() << " facet(s) generated by reconstruction." << std::endl;

    // copy points for random access
    std::vector <Point_3> vertices;
    vertices.reserve(points.size());
    std::copy(points.points().begin(), points.points().end(), std::back_inserter(vertices));

    CGAL::Surface_mesh<Point_3> lung_base_mesh;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vertices, facets, lung_base_mesh);
    // std::ofstream f("/usr/local/lib/cgal/data/lung/out_af.off");
    // f << lung_base_mesh;
    // f.close();
    CGAL::IO::write_polygon_mesh("/usr/local/lib/cgal/data/lung/ex/lung_base_mesh.ply", lung_base_mesh);

    return EXIT_SUCCESS;
}
