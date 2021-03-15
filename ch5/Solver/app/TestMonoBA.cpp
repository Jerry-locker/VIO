#include <iostream>
#include <random>
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

// Frame:保存每帧的姿态和观测
struct Frame
{
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    unordered_map<int, Eigen::Vector3d> featurePerId; //该帧观测到的特征以及特征id
};

// 产生世界坐标系下的虚拟数据:相机姿态、特征点、每帧观测
void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points)
{
    int featureNums = 20;  //特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;      //相机数目

    double radius = 8;
    for (int n = 0; n < poseNums; ++n)
    {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4圆弧
        //绕z轴旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    //随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j)
    {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        //在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i)
        {
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();  //归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);  //加上观测噪声使系统更真实
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}

int main()
{
    // 1.生成虚拟数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points);
    Eigen::Quaterniond qic(1, 0, 0, 0);
    Eigen::Vector3d tic(0, 0, 0);  //Tbc

    // 2.构建problem
    Problem problem(Problem::ProblemType::SLAM_PROBLEM);  //告诉求解器这是一个slam问题，则信息矩阵是稀疏的，便可用舒尔补的方式去加速求解(先求pose再求landmark)

    // 2-1.添加位姿顶点
    vector<shared_ptr<VertexPose> > vertexCams_vec;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        shared_ptr<VertexPose> vertexCam(new VertexPose());
        Eigen::VectorXd pose(7);  //3维平移+4维四元数  虽然顶点存的优化变量是7维的，但自由度只有6维！
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w(); //平移和四元数
        vertexCam->SetParameters(pose);

//        if(i < 2)
//        vertexCam->SetFixed();     //fix相机的操作

        problem.AddVertex(vertexCam);
        vertexCams_vec.push_back(vertexCam);
    }

    // 2-2.添加路标点逆深度顶点和边
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    vector<double> noise_invd;
    vector<shared_ptr<VertexInverseDepth> > allPoints;
    for (size_t i = 0; i < points.size(); ++i)
    {
        //虚构出路标点在第1帧相机坐标系下的逆深度(并且我们只要优化在第1帧相机坐标系下的逆深度即可)
        Eigen::Vector3d Pw = points[i];
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise);
//        double inverse_depth = 1. / Pc.z();
        noise_invd.push_back(inverse_depth);

        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        VecX inv_d(1);
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d);
        problem.AddVertex(verterxPoint);
        allPoints.push_back(verterxPoint);

        //计算每个路标点对应的投影误差(将路标点由第1帧相机坐标系转换至第2帧和第3帧相机坐标系)
        for (size_t j = 1; j < cameras.size(); ++j)
        {
            Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second;
            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
            edge->SetTranslationImuFromCamera(qic, tic);

            std::vector<std::shared_ptr<Vertex> > edge_vertex;
            edge_vertex.push_back(verterxPoint);      //优化变量1：路标点在第1帧下的逆深度
            edge_vertex.push_back(vertexCams_vec[0]); //优化变量2：第1帧相机位姿
            edge_vertex.push_back(vertexCams_vec[j]); //优化变量3：第j+1帧相机位姿(第2帧或第3帧)
            edge->SetVertex(edge_vertex);

            problem.AddEdge(edge);
        }
    }

    // 3.优化求解
    problem.Solve(5); //一共迭代5次 顶点和边准备好了直接solve即可

    //输出真实值、加了噪声的值、优化后的值  可见优化后的值与真实值十分接近   （点很接近 平移有偏移量）
    std::cout << std::endl;

    std::cout<<"------------ inverse depth ----------------"<<std::endl;
    for (size_t k = 0; k < allPoints.size(); k+=1)
    {
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
                  << noise_invd[k] << " ,opt " << allPoints[k]->Parameters() << std::endl;
    }

    std::cout<<"------------ pose translation ----------------"<<std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i)
    {
        std::cout<<"translation after opt: "<< i <<" :"<< vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: "<<cameras[i].twc.transpose()<<std::endl;
    }
    /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。 (第一帧带头大哥偏了)
    /// 解决办法： fix 第一帧和第二帧（令残差对一二帧的雅克比为0），固定 7 自由度。 或者加上非常大的先验值。

    problem.TestMarginalize();  //测试边缘化是否正确 marginalization的核心在于矩阵块操作

    return 0;
}

