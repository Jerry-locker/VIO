//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#define svd_d 1

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;  //观测
};

int main()
{
    // 1.生成10个相机位姿
    int poseNums = 10;
    double radius = 8;
    //double fx = 1.;
    //double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n )
    {
        double theta = n * 2 * M_PI / ( poseNums * 4);
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 2.随机数生成1个世界坐标系下的三维点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);
    Eigen::Vector3d Pw(tx, ty, tz);

    // 3.生成观测
    //i=3 ～ i=9观测到了该三维点
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i)
    {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z);
    }
    
    // 4.三角化(用多组观测恢复路标点的世界坐标 深度是副产品)
    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();
    /* your code begin */
    //计算D矩阵(首先需要确定矩阵维度，否则无法进行矩阵块操作！)
    int size = (end_frame_id - start_frame_id)*2;      //D矩阵行维度
    Eigen::MatrixXd D(Eigen::MatrixXd::Zero(size,4));  //D矩阵初始化
    for (int i = start_frame_id; i < end_frame_id; ++i)
    {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d tcw = -camera_pose[i].Rwc.transpose()*camera_pose[i].twc;
        //Eigen::MatrixXd T_tmp(3,4);
        Eigen::MatrixXd T_tmp(Eigen::MatrixXd::Zero(3,4));
        T_tmp.block(0,0,3,3) = Rcw;
        T_tmp.block(0,3,3,1) = tcw;
        auto Pk_1 = T_tmp.block(0,0,1,4);
        auto Pk_2 = T_tmp.block(1,0,1,4);
        auto Pk_3 = T_tmp.block(2,0,1,4);
        int j = i - start_frame_id;
        D.block(2*j, 0, 1, 4) = Pk_3*camera_pose[i].uv(0) - Pk_1;
        D.block(2*j+1, 0, 1, 4) = Pk_3*camera_pose[i].uv(1) - Pk_2;
    }

#ifdef svd_d
    //法一：对D进行SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullU|Eigen::ComputeFullV);
    //Eigen::Matrix4d V = svd.matrixV();
    //Eigen::MatrixXd V = svd.matrixV();
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto sigma = svd.singularValues(); //对角线元素
    P_est << V(0,3)/V(3,3), V(1,3)/V(3,3), V(2,3)/V(3,3);
    std::cout << "验证三角化结果：\n" << "ratio = sigma4 / sigma3 = " << sigma[3]/sigma[2] <<std::endl;
#else
    //法二：对DTD进行SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D.transpose()*D, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix4d V = svd.matrixV();
    P_est << V(0,3)/V(3,3), V(1,3)/V(3,3), V(2,3)/V(3,3);
#endif
    /* your code end */
    
    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
    return 0;
}
