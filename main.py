import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def test():
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(
        # "/Users/ethan/Downloads/max_z_processed.ply"
        # "/Users/ethan/Downloads/max_z.ply"
        "/Users/ethan/Downloads/interpolated_contact_surface.ply"
        # "/Users/ethan/Projects/diaphragm-evaluation/result/10003382/in/lung.ply"
        # "/Users/ethan/Projects/diaphragm-evaluation/result/10003382/in/thorax.ply"
        # "/Users/ethan/Projects/diaphragm-evaluation/result/10003382/in/lung_dilated_mask.ply"
    )
    print(pcd)
    o3d.visualization.draw_geometries([pcd])

    print("Statistical outlier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.0001)
    display_inlier_outlier(pcd, ind)
    o3d.visualization.draw_geometries([cl])
    o3d.io.write_point_cloud('/Users/ethan/Downloads/max_z_outlier_removed.ply', cl)

    # print("Radius outlier removal")
    # cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=6)
    # display_inlier_outlier(pcd, ind)
    # o3d.visualization.draw_geometries([cl])


if __name__ == '__main__':
    test()
