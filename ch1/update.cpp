#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.h>

using namespace std;

int main(int agrc, char** argv)
{
    //初始化R和q
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    Eigen::Quaterniond q(R);

    //更新R并输出
    Sophus::SO3 SO3_R(R);
    Eigen::Vector3d update_so3(0.01, 0.02, 0.03);
    Sophus::SO3 SO3_updated = SO3_R*Sophus::SO3::exp(update_so3);
    cout<<"R updated = "<<endl<<SO3_updated.matrix()<<endl;

    //更新q并输出
    Eigen::Quaterniond update_q(1, 0.01/2, 0.02/2, 0.03/2);
    update_q.normalize();
    Eigen::Quaterniond q_updated = q*update_q;
    cout<<"q updated = "<<endl<<q_updated.matrix()<<endl;

    //计算偏差
    Eigen::Matrix3d R_diff = SO3_updated.matrix() - q_updated.matrix();
    cout<<"diff = "<<endl<<R_diff<<endl;

    return 0;
}
