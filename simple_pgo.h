#pragma once
#include "commons.h"
#include "scan_context.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include "geometry_msgs/PoseStamped.hpp"
#include <map>
#include <string>

// LocationConstraintObs lives in commons.h (shared with CloudWithPose).

struct KeyPoseWithCloud
{
    M3D r_local;
    V3D t_local;
    M3D r_global;
    V3D t_global;
    double time;
    CloudType::Ptr body_cloud;
};
struct LoopPair
{
    size_t source_id;
    size_t target_id;
    M3D r_offset;
    V3D t_offset;
    double score;                       // diagnostic: ICP fitness (lidar) or 0 otherwise
    gtsam::SharedNoiseModel noise;      // per-constraint noise model (set at detection)
};

struct Config
{
    double key_pose_delta_deg = 10;
    double key_pose_delta_trans = 1.0;
    double loop_search_radius = 1.0;
    double loop_time_thresh = 60.0;
    double loop_score_thresh = 0.15;
    int loop_submap_half_range = 5;
    double submap_resolution = 0.1;
    double min_loop_detect_duration = 10.0;
    // Sanity gate: skip ICP if candidate keyframe is farther than this
    // from current keyframe in global pose. 0 disables the check.
    double loop_candidate_max_distance_m = 30.0;

    // Feature-poverty gate: skip loop search when the current scan's
    // descriptor vertical-structure std (scan_context::descriptor_structure)
    // is below this — the scan can't reliably place itself (open grass field),
    // so any closure would be noise. 0 disables (default = current behavior).
    // Superseded in practice by loop_min_occupancy + loop_min_degeneracy below
    // (structure overlaps too much between scenes to threshold cleanly).
    double min_descriptor_std = 0.0;

    // Structure-spread gate: require at least this many occupied Scan-Context
    // cells. Open grass returns cluster near the sensor (few rings filled);
    // built environments spread returns out to range. Calibrated on go2 fastlio
    // (1200-cell 20x60 descriptor): grassy ~70 vs gir_park ~88 vs downtown ~120
    // at the SAME point count, so this measures spread, not density. 0 disables.
    int loop_min_occupancy = 80;

    // Observability gate (Zhang 2016 / X-ICP degeneracy factor): reject a loop
    // candidate whose source scan's smallest normalized normal-scatter
    // eigenvalue is below this. A planar/degenerate scan (open grass) -> ~0:
    // ICP slides freely in-plane and reports low fitness for a bogus closure.
    // Real scenes (incl. sparse gir_park) sit >0.15; grassy's firing closures
    // sit ~0.01. 0 disables.
    double loop_min_degeneracy = 0.05;

    // When true, log one "PGO_DIAG ..." line per loop candidate that reaches
    // ICP (accepted OR rejected) with its fitness / candidate distance /
    // descriptor structure — the data to design the right loop-acceptance gate.
    bool debug = false;

    // Robust (M-estimator) kernel wrapping *all* loop factors (lidar + landmark).
    // Off by default to preserve the original Gaussian behavior bit-for-bit.
    bool loop_robust_kernel = false;
    double loop_robust_huber_k = 1.345;

    // --- Location constraints (decoupled constraint-event ingestion) ---------
    // When true, the PGO ingests LocationConstraint events (a separate perceiver
    // emits them). Each becomes its own pose node (placed from interpolated
    // odometry at the constraint's timestamp) plus a BetweenFactor(node,
    // location) whose noise model is the covariance carried in the message. Two
    // constraints sharing a to_id observe the same graph variable -> GTSAM
    // closes the loop automatically. Off by default.
    bool use_location_constraints = false;

    // --- Odometry between-factor noise (anisotropic) -------------------------
    // The LIO front end's *relative* roll/pitch between consecutive keyframes is
    // accurate (the IMU sees gravity each step), but heading (yaw) drifts. So the
    // odometry constraint is anisotropic: stiff roll/pitch, looser yaw. This is
    // what keeps a loop closure from sloshing its (large, mostly-yaw) correction
    // into roll/pitch — a tilt that would convert horizontal travel into vertical
    // and corrupt z by tens of metres. Because the roll/pitch variance is small
    // but NONZERO, accumulated tilt drift can still be corrected slowly by
    // landmarks across the graph (it is "accurate but not perfect", not a lock).
    double odom_rot_rp_var = 1e-8;    // roll/pitch (accurate relative tilt)
    double odom_rot_yaw_var = 1e-5;   // yaw (heading drifts more)
    double odom_trans_xy_var = 1e-4;  // x, y
    double odom_trans_z_var = 1e-6;   // z

    // --- Gravity anchor ------------------------------------------------------
    // The first keyframe's orientation is gravity-aligned by the LIO front end
    // (initial roll/pitch is correct). The anchor is a PriorFactor on keyframe 0
    // that pins it so the optimizer (pulled by landmark/loop factors) cannot
    // rotate the initial pitch/roll away from gravity. The full pose is pinned
    // (it is also the gauge reference); the roll/pitch stiffness is the gravity
    // component and is exposed separately. Variances (smaller = stiffer).
    bool gravity_anchor = true;
    double gravity_anchor_rp_var = 1e-12;     // roll/pitch (the gravity lock)
    double gravity_anchor_yaw_var = 1e-12;    // yaw (gauge)
    double gravity_anchor_trans_var = 1e-12;  // translation (gauge)

    // Per-keyframe gravity anchor. Anchoring gravity only on keyframe 0 is not
    // enough: a large loop closure (correcting accumulated yaw) is free to slosh
    // some of the correction into the *inner* keyframes' roll/pitch, because
    // nothing else pins them — and a tilt converts horizontal travel into
    // vertical, so the corrected z drifts wildly (tens of metres) even though the
    // LIO front end's roll/pitch are gravity-accurate. This adds a roll/pitch-only
    // PriorFactor on EVERY keyframe (target = the keyframe's gravity-aligned LIO
    // orientation; yaw + translation left free for loop closure), so the closure
    // corrects yaw/x-y only and the gravity-correct z structure is preserved.
    // Default OFF: the anisotropic odometry between-factor (above) is the primary
    // gravity-preservation mechanism and lets landmarks still correct slow tilt
    // drift. This absolute prior is a harder lock — useful once the front end's
    // absolute tilt is made trustworthy (e.g. by ZUPT in the LIO estimator).
    bool gravity_anchor_per_keyframe = false;
    double gravity_anchor_kf_rp_var = 1e-4;   // (~0.57 deg) roll/pitch stiffness

    // Scan Context settings
    bool use_scan_context = true;
    int scan_context_num_rings = 20;
    int scan_context_num_sectors = 60;
    double scan_context_max_range_m = 80.0;
    int scan_context_top_k = 10;
    double scan_context_match_threshold = 0.4;
    double scan_context_lidar_height_m = 2.0;
};

class SimplePGO
{
public:
    SimplePGO(const Config &config);

    bool isKeyPose(const PoseWithTime &pose);

    bool addKeyPose(const CloudWithPose &cloud_with_pose);

    // Ingest a LocationConstraint. Adds its own pose node at `pose` (the caller
    // interpolates odometry at the constraint's timestamp; `frame_id` is that
    // odometry's frame) linked to the backbone by an odom between-factor, then a
    // BetweenFactor(node, location) using the constraint's covariance. Returns
    // false (and does nothing) if no keyframe exists yet to anchor from.
    bool addLocationConstraint(const PoseWithTime &pose, const std::string &frame_id,
                               const LocationConstraintObs &constraint);

    bool hasLoop(){return m_cache_pairs.size() > 0;}

    void searchForLoopPairs();

    void smoothAndUpdate();

    CloudType::Ptr getSubMap(int idx, int half_range, double resolution);
    std::vector<std::pair<size_t, size_t>> &historyPairs() { return m_history_pairs; }
    std::vector<KeyPoseWithCloud> &keyPoses() { return m_key_poses; }

    M3D offsetR() { return m_r_offset; }
    V3D offsetT() { return m_t_offset; }

    // Parallel to GTSAM's Values (key -> Pose3): our key -> PoseStamped, carrying
    // the frame_id each node's pose was created in (from the odometry). GTSAM
    // nodes have no frame; this is where we keep that bookkeeping. (Not consumed
    // yet — scaffolding for future automatic cross-frame node additions.)
    const std::map<gtsam::Key, geometry_msgs::PoseStamped> &nodePoses() const { return m_node_poses; }

    // Place recognition exposed for diagnostics / persistence.
    const std::vector<scan_context::Descriptor>& descriptors() const { return m_scan_context_descriptors; }
    const std::vector<scan_context::RingKey>& ringKeys() const { return m_scan_context_ring_keys; }

private:
    // Scan-context-based candidate search; returns -1 if no acceptable match.
    // out_best / out_second report the closest and 2nd-closest candidate cosine
    // distances (for the Lowe ratio distinctiveness signal).
    int searchByScanContext(int& out_sector_shift, float& out_best, float& out_second) const;
    // Original position-based fallback (radius search on past key-pose
    // positions). Kept for ablation + when scan context is disabled.
    int searchByPosition() const;

    // Insert a new pose node (key = next contiguous index) at `pose`: initial
    // value + a backbone factor (gravity prior on the first, else an odom
    // between-factor to the previous node), the optional per-keyframe gravity
    // anchor, the scan-context cache (empty when cloud is null), and the
    // key -> PoseStamped frame record. Returns the new node's index.
    size_t insertPoseNode(const PoseWithTime &pose, CloudType::Ptr cloud,
                          const std::string &frame_id);

    // Constraint factor for a freshly-inserted node: ensure a graph variable for
    // to_id exists (initialized from this node's global pose), add a
    // BetweenFactor(node, location) with the constraint's covariance, and apply
    // constraint_instance_id revision by scheduling removal of any committed
    // factors with the same instance id. Stages the new factor's graph position
    // so its iSAM2 factor index can be captured after the update.
    void addLocationConstraintFactors(size_t node_idx, const LocationConstraintObs &constraint);

    Config m_config;
    scan_context::Config m_scan_context_config;
    std::vector<KeyPoseWithCloud> m_key_poses;
    std::vector<std::pair<size_t, size_t>> m_history_pairs;
    std::vector<LoopPair> m_cache_pairs;
    std::vector<scan_context::Descriptor> m_scan_context_descriptors;
    std::vector<scan_context::RingKey> m_scan_context_ring_keys;
    M3D m_r_offset;
    V3D m_t_offset;
    std::shared_ptr<gtsam::ISAM2> m_isam2;
    gtsam::Values m_initial_values;
    gtsam::NonlinearFactorGraph m_graph;
    pcl::IterativeClosestPoint<PointType, PointType> m_icp;

    // key -> PoseStamped (frame bookkeeping; see nodePoses()).
    std::map<gtsam::Key, geometry_msgs::PoseStamped> m_node_poses;

    // --- Location-constraint bookkeeping -------------------------------------
    // to_id -> the location's variable index (gtsam key = Symbol('l', index)).
    std::map<std::string, int> m_location_index;
    int m_next_location = 0;
    // A constraint factor staged this update cycle, with its position in m_graph
    // so its assigned iSAM2 factor index can be captured afterwards.
    struct StagedConstraintFactor { std::string instance_id; size_t graph_pos; };
    std::vector<StagedConstraintFactor> m_staged_constraint_factors;
    // constraint_instance_id -> committed iSAM2 factor indices, so a revised
    // constraint reusing the same instance id can remove the superseded factors.
    std::map<std::string, std::vector<size_t>> m_committed_by_instance;
    // Factor indices to remove on the next iSAM2 update (revision).
    std::vector<size_t> m_pending_removals;
    // Set when a constraint closes a loop (re-sights an existing location) or a
    // revision removes factors this cycle, so smoothAndUpdate runs the extra
    // relinearization passes a large correction needs (mirrors the lidar path).
    bool m_location_closure = false;
};
