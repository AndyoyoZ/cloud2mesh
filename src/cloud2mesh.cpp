//点的类型的头文件
#include <pcl/point_types.h>
//点云文件IO（pcd文件和ply文件）
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
//kd树
#include <pcl/kdtree/kdtree_flann.h>
//特征提取
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
//重构
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
//可视化
#include <pcl/visualization/pcl_visualizer.h>
//多线程
#include <boost/thread/thread.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
 
int main(int argc, char** argv)
{
	// 确定文件格式
	char tmpStr[100];
	strcpy(tmpStr, argv[1]);
	char* pext = strrchr(tmpStr, '.');
	std::string extply("ply");
	std::string extpcd("pcd");
	if (pext){
		*pext = '\0';
		pext++;
	}
	std::string ext(pext);
	//如果不支持文件格式，退出程序
	if (!((ext == extply) || (ext == extpcd))){
		std::cout << "文件格式不支持!" << std::endl;
		std::cout << "支持文件格式：*.pcd和*.ply！" << std::endl;
		return(-1);
	}
 
	//根据文件格式选择输入方式
	pcl::PointCloud < pcl::PointXYZRGB > ::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>); //创建点云对象指针，用于存储输入
	if (ext == extply){
		if (pcl::io::loadPLYFile(argv[1], *cloud) == -1){
			PCL_ERROR("Could not read ply file!\n");
			return -1;
		}
	}
	else{
		if (pcl::io::loadPCDFile(argv[1], *cloud) == -1){
			PCL_ERROR("Could not read pcd file!\n");
			return -1;
		}
	}
 
	// 计算法向量
 
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;//法线估计对象
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);//存储估计的法线的指针
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals); //计算法线，结果存储在normals中
 
	//将点云和法线放到一起
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
 
	//创建搜索树
	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree2->setInputCloud(cloud_with_normals);
	//创建Poisson对象，并设置参数
	pcl::Poisson<pcl::PointXYZRGBNormal> pn;
	pn.setConfidence(true); //是否使用法向量的大小作为置信信息。如果false，所有法向量均归一化。
	pn.setDegree(2); //设置参数degree[1,5],值越大越精细，耗时越久。
	pn.setDepth(8); //树的最大深度，求解2^d x 2^d x 2^d立方体元。由于八叉树自适应采样密度，指定值仅为最大深度。
	pn.setIsoDivide(8); //用于提取ISO等值面的算法的深度
	pn.setManifold(true); //是否添加多边形的重心，当多边形三角化时。 设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
	pn.setOutputPolygons(false); //是否输出多边形网格（而不是三角化移动立方体的结果）
	pn.setSamplesPerNode(3.0); //设置落入一个八叉树结点中的样本点的最小数量。无噪声，[1.0-5.0],有噪声[15.-20.]平滑
	pn.setScale(1.25); //设置用于重构的立方体直径和样本边界立方体直径的比率。
	pn.setSolverDivide(8); //设置求解线性方程组的Gauss-Seidel迭代方法的深度
	//pn.setIndices();
 
	//设置搜索方法和输入点云
	pn.setSearchMethod(tree2);
	pn.setInputCloud(cloud_with_normals);
	//创建多变形网格，用于存储结果
	pcl::PolygonMesh mesh;
	//执行重构
	pn.performReconstruction(mesh);

    //给mesh染色
    pcl::PointCloud<pcl::PointXYZRGB> cloud_color_mesh;
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
                // cout<<"dis:"<<dist<<endl;
            }
        }
        // cloud_color_mesh.points[i].r = int(red/pointIdxNKNSearch.size ()+0.5);
        // cloud_color_mesh.points[i].g = int(green/pointIdxNKNSearch.size ()+0.5);
        // cloud_color_mesh.points[i].b = int(blue/pointIdxNKNSearch.size ()+0.5);

        cloud_color_mesh.points[i].r = 255;
        cloud_color_mesh.points[i].g = 0;
        cloud_color_mesh.points[i].b = 0;
    }
    pcl::toPCLPointCloud2(cloud_color_mesh, mesh.cloud);




 
	//保存网格图
	pcl::io::savePLYFile("../data/result.ply", mesh);
 
	// 显示结果图
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(mesh, "my");
	viewer->addCoordinateSystem(50.0);
	viewer->initCameraParameters();
	while (!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
 
	return 0;
}