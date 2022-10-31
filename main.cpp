#define CGAL_USE_BASIC_VIEWER

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/draw_surface_mesh.h>
#include <CGAL/polygon_mesh_processing.h>

#include <iostream>
#include <cmath>
#include <filesystem>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " /root/diaphragm-evaluation/result/10003382/ex "
                     "10.lung_diaphragm_contact_surface_mesh(manually).ply"
                  << std::endl;
    }

    std::filesystem::path result_dir_path(argv[1]);
    std::filesystem::path mesh_filename(argv[2]);
    std::filesystem::path mesh_file_path = result_dir_path / mesh_filename;

    std::cout << "Reading mesh file: " << mesh_file_path << std::endl;

    Mesh lung_diaphragm_contact_surface_mesh;

    if (!CGAL::IO::read_polygon_mesh(mesh_file_path, lung_diaphragm_contact_surface_mesh)) {
        std::cerr << "Lung Diaphragm Contact Surface Mesh Invalid Input File." << std::endl;
        return EXIT_FAILURE;
    }
    auto area = CGAL::Polygon_mesh_processing::area(lung_diaphragm_contact_surface_mesh);

    std::ofstream log_file;
    log_file.open("/root/ct/log.csv", std::ios::app);
    log_file << result_dir_path << "," << area << "\n";

    std::cout << CGAL::Polygon_mesh_processing::area(lung_diaphragm_contact_surface_mesh) << "\n";

    return 0;
}
