#include <iostream>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/poisson.h>

using namespace pcl;
using namespace std;

void poisson_reconstruction(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_cloud)
{
    cout << "begin filter ..." << endl;
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>());
    pcl::copyPointCloud(*object_cloud, *cloud); 
    PointCloud<PointXYZRGB>::Ptr filtered(new PointCloud<PointXYZRGB>());
    PassThrough<PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    filter.filter(*filtered);
    cout << "passthrough filter complete" << endl;
    cout << "begin normal estimation" << endl;
    NormalEstimationOMP<PointXYZRGB, Normal> ne;//计算点云法向
    ne.setNumberOfThreads(8);//设定临近点
    ne.setInputCloud(filtered); 
    ne.setRadiusSearch(0.01);//设定搜索半径
    Eigen::Vector4f centroid; 
    compute3DCentroid(*filtered, centroid);//计算点云中心
    ne.setViewPoint(centroid[0], centroid[1], centroid[2]);//将向量计算原点置于点云中心
    PointCloud<Normal>::Ptr cloud_normals (new PointCloud<Normal>());
    ne.compute(*cloud_normals);
    cout << "normal estimation complete" << endl;
    cout << "reverse normals' direction" << endl; 
    //将法向量反向
    for(size_t i = 0; i < cloud_normals->size(); ++i)
    {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }
    //融合RGB点云和法向
    cout << "combine points and normals" << endl;
    PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointXYZRGBNormal>());
    concatenateFields(*filtered, *cloud_normals, *cloud_smoothed_normals);
    //泊松重建
    cout << "begin poisson reconstruction" << endl;
    Poisson<PointXYZRGBNormal> poisson;
    //poisson.setDegree(2);
    poisson.setDepth(8);
    poisson.setSolverDivide (6);
    poisson.setIsoDivide (6);
    poisson.setConfidence(false);
    poisson.setManifold(false);
    poisson.setOutputPolygons(false);
    poisson.setInputCloud(cloud_smoothed_normals);
    PolygonMesh mesh;
    poisson.reconstruct(mesh);
    cout << "finish poisson reconstruction" << endl;
    //给mesh染色
    PointCloud<PointXYZRGB> cloud_color_mesh;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud_color_mesh);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud (cloud);
    // K nearest neighbor search 
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for(int i=0;i<cloud_color_mesh.points.size();++i)
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        float dist = 0.0;
        int red = 0;
        int green = 0;
        int blue = 0;
        uint32_t rgb;
        if ( kdtree.nearestKSearch (cloud_color_mesh.points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            for (int j = 0; j < pointIdxNKNSearch.size (); ++j)
            {
                r = cloud->points[ pointIdxNKNSearch[j] ].r;
                g = cloud->points[ pointIdxNKNSearch[j] ].g;
                b = cloud->points[ pointIdxNKNSearch[j] ].b;
                red += int(r); 
                green += int(g);
                blue += int(b); 
                dist += 1.0/pointNKNSquaredDistance[j];
                // std::cout<<"red: "<<int(r)<<std::endl;
                // std::cout<<"green: "<<int(g)<<std::endl;
                // std::cout<<"blue: "<<int(b)<<std::endl;
                cout<<"dis:"<<dist<<endl;
            }
        }
        cloud_color_mesh.points[i].r = int(red/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].g = int(green/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].b = int(blue/pointIdxNKNSearch.size ()+0.5);
    }
    toPCLPointCloud2(cloud_color_mesh, mesh.cloud);
    io::savePLYFile("../data/object_mesh.ply", mesh);
}



int main(int argc,char **argv)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_s(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::io::loadPCDFile("../data/test.pcd",*cloud_s);
    poisson_reconstruction(cloud_s);
    return 0;

}
