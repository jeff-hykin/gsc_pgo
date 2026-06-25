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

// A landmark sighting routed in as a decoupled Landmark *event*. Pose is the
// landmark in the body frame at the observing
// keyframe (camera->body extrinsic resolved upstream). `id` is the URL-like
// Landmark.id; two sightings sharing an id observe the same graph variable,
// which closes the loop. `replacement_ms` > 0 => corrective: any committed
// same-id factor observed within that window before `ts` is removed.
struct LandmarkObs {
    std::string id;
    M3D r_body_lm;
    V3D t_body_lm;
    double ts = 0.0;
    double confidence = 0.0;
    double replacement_ms = 0.0;
};

struct CloudWithPose {
    CloudType::Ptr cloud;
    PoseWithTime pose;
    // Landmark events associated with this scan (use_landmarks path).
    std::vector<LandmarkObs> landmark_obs;
};
