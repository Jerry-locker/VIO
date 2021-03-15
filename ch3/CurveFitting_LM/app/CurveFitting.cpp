#include <iostream>
#include <random>
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

// 顶点
class CurveFittingVertex: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex(): Vertex(3){} //abc三个参数，因此顶点实际存储的优化变量为3维，存储在Vertex的parameters_中
    virtual std::string TypeInfo() const {return "abc";}
};

// 边
class CurveFittingEdge: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x, double y): Edge(1,1,std::vector<std::string>{"abc"}) //边即残差的维度为1、连接的顶点个数为1
    {
        x_ = x;
        y_ = y;
    }

    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters(); //估计的参数
        residual_(0) = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) ) - y_; //构建残差
        //residual_(0) = abc(0)*x_*x_ + abc(1)*x_ + abc(2) - y_;
    }

    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();
        double exp_y = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) );

        Eigen::Matrix<double, 1, 3> jaco_abc; //误差为1维，优化变量为3维，所以是1x3的雅克比矩阵
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;
        //jaco_abc << x_*x_, x_, 1;
        jacobians_[0] = jaco_abc;
    }
    virtual std::string TypeInfo() const override {return "CurveFittingEdge";} //返回边的类型信息
public:
    double x_,y_;  //x值、y值为_measurement
};

int main()
{
    double a=1.0, b=2.0, c=1.0;         //真实参数值
    int N = 100;                        //数据点
    double w_sigma= 1.;                 //噪声Sigma值

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    // 1.构建最小二乘问题
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM); //是通用的最小二乘问题还是slam的最小二乘问题 这里是曲线拟合 所以是通用

    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());
    vertex->SetParameters(Eigen::Vector3d (0.,0.,0.)); //设定待估计参数 a, b, c初始值
    problem.AddVertex(vertex);

    //构造N次观测
    for (int i = 0; i < N; ++i)
    {
        double x = i/100.;
        double n = noise(generator);
        double y = std::exp( a*x*x + b*x + c ) + n;  //产生有噪声的观测
        //double y = a*x*x + b*x + c + n;  //产生有噪声的观测
        //double y = std::exp( a*x*x + b*x + c );

        //每个观测对应的残差函数
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x,y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);

        //把这个残差添加到最小二乘问题
        problem.AddEdge(edge);
    }

    // 2.使用LM求解
    std::cout<<"\nTest CurveFitting start..."<<std::endl;
    problem.Solve(30);
    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;

    return 0;
}


