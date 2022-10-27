function lung_base_gridfit(point_cloud_dir_path, point_cloud_filename)
    point_cloud_file_path = strcat(point_cloud_dir_path, '/', point_cloud_filename);
    ptCloud = pcread(point_cloud_file_path);
    x=double(ptCloud.Location(:,1));
    y=double(ptCloud.Location(:,2));
    z=double(ptCloud.Location(:,3));

    gx=0:3:400;
    gy=0:3:400;
    g=gridfit(x,y,z,gx,gy);

    writematrix(g,strcat(point_cloud_dir_path, '/', '9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_g.csv'));
    writematrix(gx,strcat(point_cloud_dir_path, '/', '9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_gx.csv'));
    writematrix(gy,strcat(point_cloud_dir_path, '/', '9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_gy.csv'));

%     plot3(x,y,z,'.');
%     figure;
%     colormap(hot(256));
%     surf(gx,gy,g);
%     camlight right;
%     lighting phong;
%     shading interp;
%     line(x,y,z,'marker','.','markersize',4,'linestyle','none');
%     title 'Use topographic contours to recreate a surface';
end
