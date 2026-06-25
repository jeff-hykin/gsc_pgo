#pragma once
#include <Eigen/Eigen>
#include <map>
#include <string>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using PointType = pcl::PointXYZI;
using CloudType = pcl::PointCloud<PointType>;
using PointVec = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

using M3D = Eigen::Matrix3d;
using V3D = Eigen::Vector3d;
using M6D = Eigen::Matrix<double, 6, 6>;
using M3F = Eigen::Matrix3f;
using V3F = Eigen::Vector3f;
using M4F = Eigen::Matrix4f;
using V4F = Eigen::Vector4f;

struct PoseWithTime {
    V3D t;
    M3D r;
    int32_t sec;
    uint32_t nsec;
    double second;
    void setTime(int32_t sec, uint32_t nsec);
};

// A location constraint routed in as a decoupled event: a relative-pose
// measurement from the body frame to some location variable, plus its 6x6
// covariance — i.e. ready-made args for a GTSAM BetweenFactor. `to_id` is the
// location variable identity (URL-like); two constraints sharing a to_id observe
// the same graph variable, which closes the loop. `constraint_instance_id`
// identifies this specific external instance, so a later revision can remove all
// committed factors with the same instance id (rolling outlier removal).
struct LocationConstraintObs {
    std::string to_id;
    std::string constraint_instance_id;
    M3D r_body_loc;   // rotation of the transform body -> location
    V3D t_body_loc;   // translation of the transform body -> location
    M6D covariance;   // 6x6, GTSAM Pose3 tangent order [rot(3), trans(3)]
    double ts = 0.0;
};

struct CloudWithPose {
    CloudType::Ptr cloud;
    PoseWithTime pose;
    std::string frame_id;   // odometry header.frame_id (the frame `pose` is in)
};
