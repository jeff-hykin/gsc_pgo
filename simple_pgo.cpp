#include "simple_pgo.h"

#include <cstdio>
#include <limits>
#include <pcl/features/normal_3d.h>

namespace {
// Geometric degeneracy of a scan, à la Zhang 2016 ("On Degeneracy of
// Optimization-based State Estimation") / X-ICP. Build the point-to-plane
// translational information matrix M = Σ nᵢ nᵢᵀ over surface normals; its
// eigenvalues say how constrained the alignment is along each axis. A flat
// open field has all normals ≈ vertical → M is rank-1 → the smallest
// normalized eigenvalue → 0 → translation slides freely in-plane (exactly why
// ICP fitness lies on grass). Walls facing several directions fill all three
// eigenvalues. Eigenvalues of Σ nnᵀ are invariant to global frame rotation,
// so the (global-frame) submap is fine. Writes the two smaller normalized
// eigenvalues (e_min ≤ e_mid ≤ e_max, summing to 1) to the out-params.
void cloud_degeneracy(const CloudType::Ptr& cloud, float& e_min, float& e_mid)
{
    e_min = -1.0f;
    e_mid = -1.0f;
    if (!cloud || cloud->size() < 20)
        return;
    pcl::NormalEstimation<PointType, pcl::Normal> normal_estimator;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(10);
    normal_estimator.compute(*normals);

    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    long valid = 0;
    for (const auto& normal : normals->points) {
        if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.normal_y) ||
            !std::isfinite(normal.normal_z))
            continue;
        Eigen::Vector3d n(normal.normal_x, normal.normal_y, normal.normal_z);
        scatter += n * n.transpose();
        valid++;
    }
    if (valid < 20)
        return;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();  // ascending
    const double trace = eigenvalues.sum();
    if (trace <= 0.0)
        return;
    e_min = static_cast<float>(eigenvalues(0) / trace);
    e_mid = static_cast<float>(eigenvalues(1) / trace);
}
}  // namespace

SimplePGO::SimplePGO(const Config &config) : m_config(config)
{
    gtsam::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.01;
    isam2_params.relinearizeSkip = 1;
    m_isam2 = std::make_shared<gtsam::ISAM2>(isam2_params);
    m_initial_values.clear();
    m_graph.resize(0);
    m_r_offset.setIdentity();
    m_t_offset.setZero();

    m_icp.setMaximumIterations(50);
    m_icp.setMaxCorrespondenceDistance(10);
    m_icp.setTransformationEpsilon(1e-6);
    m_icp.setEuclideanFitnessEpsilon(1e-6);
    m_icp.setRANSACIterations(0);

    m_scan_context_config.n_rings = m_config.scan_context_num_rings;
    m_scan_context_config.n_sectors = m_config.scan_context_num_sectors;
    m_scan_context_config.max_range_m = m_config.scan_context_max_range_m;
    m_scan_context_config.candidate_top_k = m_config.scan_context_top_k;
    m_scan_context_config.match_threshold = m_config.scan_context_match_threshold;
    m_scan_context_config.lidar_height_m = m_config.scan_context_lidar_height_m;
}

bool SimplePGO::isKeyPose(const PoseWithTime &pose)
{
    if (m_key_poses.size() == 0)
        return true;
    const KeyPoseWithCloud &last_item = m_key_poses.back();
    double delta_trans = (pose.t - last_item.t_local).norm();
    double delta_deg = Eigen::Quaterniond(pose.r).angularDistance(Eigen::Quaterniond(last_item.r_local)) * 57.324;
    if (delta_trans > m_config.key_pose_delta_trans || delta_deg > m_config.key_pose_delta_deg)
        return true;
    return false;
}
static geometry_msgs::PoseStamped make_pose_stamped(const M3D &r, const V3D &t,
                                                    const std::string &frame_id, double ts)
{
    geometry_msgs::PoseStamped ps;
    ps.header.frame_id = frame_id;
    ps.header.stamp.sec = static_cast<int32_t>(ts);
    ps.header.stamp.nsec = static_cast<uint32_t>((ts - static_cast<int32_t>(ts)) * 1e9);
    Eigen::Quaterniond q(r);
    ps.pose.position.x = t.x();
    ps.pose.position.y = t.y();
    ps.pose.position.z = t.z();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    return ps;
}

size_t SimplePGO::insertPoseNode(const PoseWithTime &pose, CloudType::Ptr cloud,
                                 const std::string &frame_id)
{
    size_t idx = m_key_poses.size();
    M3D init_r = m_r_offset * pose.r;
    V3D init_t = m_r_offset * pose.t + m_t_offset;
    // 添加初始值
    m_initial_values.insert(idx, gtsam::Pose3(gtsam::Rot3(init_r), gtsam::Point3(init_t)));
    if (idx == 0)
    {
        // Absolute anchor prior on the first keyframe (always present: fixes the
        // pose-graph gauge). Pose3 tangent order is [rot(3), trans(3)]; the first
        // two rotation components are roll/pitch, the third is yaw. A tight
        // anchor_rp_var pins roll/pitch to the initial (LIO, gravity-aligned)
        // attitude so loop/landmark factors cannot tilt the map; a loose one frees
        // it. Yaw + translation stay stiff, which is what fixes the gauge.
        gtsam::Vector6 prior_var;
        prior_var << m_config.anchor_rp_var,
                     m_config.anchor_rp_var,
                     m_config.anchor_yaw_var,
                     m_config.anchor_trans_var,
                     m_config.anchor_trans_var,
                     m_config.anchor_trans_var;
        gtsam::noiseModel::Diagonal::shared_ptr noise =
            gtsam::noiseModel::Diagonal::Variances(prior_var);
        m_graph.add(gtsam::PriorFactor<gtsam::Pose3>(idx, gtsam::Pose3(gtsam::Rot3(init_r), gtsam::Point3(init_t)), noise));
    }
    else
    {
        // 添加里程计约束
        const KeyPoseWithCloud &last_item = m_key_poses.back();
        M3D r_between = last_item.r_local.transpose() * pose.r;
        V3D t_between = last_item.r_local.transpose() * (pose.t - last_item.t_local);
        // Anisotropic: stiff relative roll/pitch (gravity-accurate), looser yaw.
        gtsam::noiseModel::Diagonal::shared_ptr noise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << m_config.odom_rot_rp_var, m_config.odom_rot_rp_var,
             m_config.odom_rot_yaw_var, m_config.odom_trans_xy_var,
             m_config.odom_trans_xy_var, m_config.odom_trans_z_var).finished());
        m_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(idx - 1, idx, gtsam::Pose3(gtsam::Rot3(r_between), gtsam::Point3(t_between)), noise));

        // Per-keyframe roll/pitch prior: pin this keyframe's roll/pitch to its
        // initial LIO orientation (init_r), leaving yaw + translation free (huge
        // variance). Keeps a loop closure from tilting inner keyframes and
        // corrupting z (see per_keyframe_rp_prior in simple_pgo.h).
        if (m_config.per_keyframe_rp_prior)
        {
            gtsam::Vector6 grav_var;
            grav_var << m_config.per_keyframe_rp_var,
                        m_config.per_keyframe_rp_var,
                        1e8, 1e8, 1e8, 1e8;  // yaw + translation unconstrained
            gtsam::noiseModel::Diagonal::shared_ptr grav_noise =
                gtsam::noiseModel::Diagonal::Variances(grav_var);
            m_graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                idx, gtsam::Pose3(gtsam::Rot3(init_r), gtsam::Point3(init_t)), grav_noise));
        }
    }
    KeyPoseWithCloud item;
    item.time = pose.second;
    item.r_local = pose.r;
    item.t_local = pose.t;
    item.body_cloud = cloud;
    item.r_global = init_r;
    item.t_global = init_t;
    m_key_poses.push_back(item);

    // Record this node's frame (from the odometry) alongside its pose. GTSAM's
    // own Pose3 value carries no frame; this is the parallel bookkeeping.
    m_node_poses[idx] = make_pose_stamped(pose.r, pose.t, frame_id, pose.second);

    // Cache the Scan Context descriptor + ring-key for this node (empty when the
    // node has no cloud, e.g. a constraint-triggered node).
    if (cloud) {
        scan_context::Descriptor descriptor =
            scan_context::make_descriptor(*cloud, m_scan_context_config);
        m_scan_context_ring_keys.push_back(scan_context::make_ring_key(descriptor));
        m_scan_context_descriptors.push_back(std::move(descriptor));
    } else {
        m_scan_context_descriptors.emplace_back();
        m_scan_context_ring_keys.emplace_back();
    }

    return idx;
}

bool SimplePGO::addKeyPose(const CloudWithPose &cloud_with_pose)
{
    if (!isKeyPose(cloud_with_pose.pose))
        return false;
    insertPoseNode(cloud_with_pose.pose, cloud_with_pose.cloud, cloud_with_pose.frame_id);
    return true;
}

bool SimplePGO::addLocationConstraint(const PoseWithTime &pose, const std::string &frame_id,
                                      const LocationConstraintObs &constraint)
{
    // Each LocationConstraint becomes its OWN pose node (placed at the
    // interpolated-odometry pose for its timestamp), linked to the backbone by an
    // odom between-factor — so the constraint's transform is used as-is, with no
    // re-basing, and the odometry uncertainty is modeled by that backbone factor.
    // A new node needs a previous node to bridge from; a constraint arriving
    // before any keyframe exists would become the graph origin with no odometry
    // context, so drop it instead.
    // NOTE: every constraint adds a node, so rapid revision bursts can grow the
    // graph quickly. Likely a micro-optimization; revisit (dedup by instance id /
    // reuse the node) only if node count becomes a problem.
    if (m_key_poses.empty())
        return false;
    size_t node_idx = insertPoseNode(pose, nullptr, frame_id);
    addLocationConstraintFactors(node_idx, constraint);
    return true;
}

CloudType::Ptr SimplePGO::getSubMap(int idx, int half_range, double resolution)
{
    assert(idx >= 0 && idx < static_cast<int>(m_key_poses.size()));
    int min_idx = std::max(0, idx - half_range);
    int max_idx = std::min(static_cast<int>(m_key_poses.size()) - 1, idx + half_range);

    CloudType::Ptr ret(new CloudType);
    for (int i = min_idx; i <= max_idx; i++)
    {

        CloudType::Ptr body_cloud = m_key_poses[i].body_cloud;
        CloudType::Ptr global_cloud(new CloudType);
        pcl::transformPointCloud(*body_cloud, *global_cloud, m_key_poses[i].t_global, Eigen::Quaterniond(m_key_poses[i].r_global));
        *ret += *global_cloud;
    }
    if (resolution > 0)
    {
        pcl::VoxelGrid<PointType> voxel_grid;
        voxel_grid.setLeafSize(resolution, resolution, resolution);
        voxel_grid.setInputCloud(ret);
        voxel_grid.filter(*ret);
    }
    return ret;
}

int SimplePGO::searchByPosition() const
{
    size_t cur_idx = m_key_poses.size() - 1;
    const KeyPoseWithCloud &last_item = m_key_poses.back();
    pcl::PointXYZ last_pose_pt;
    last_pose_pt.x = last_item.t_global(0);
    last_pose_pt.y = last_item.t_global(1);
    last_pose_pt.z = last_item.t_global(2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr key_poses_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < cur_idx; i++)
    {
        pcl::PointXYZ pt;
        pt.x = m_key_poses[i].t_global(0);
        pt.y = m_key_poses[i].t_global(1);
        pt.z = m_key_poses[i].t_global(2);
        key_poses_cloud->push_back(pt);
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(key_poses_cloud);
    std::vector<int> ids;
    std::vector<float> sqdists;
    int neighbors = kdtree.radiusSearch(last_pose_pt, m_config.loop_search_radius, ids, sqdists);
    if (neighbors == 0)
        return -1;

    for (size_t i = 0; i < ids.size(); i++)
    {
        int idx = ids[i];
        if (std::abs(last_item.time - m_key_poses[idx].time) > m_config.loop_time_thresh)
        {
            return idx;
        }
    }
    return -1;
}

int SimplePGO::searchByScanContext(int& out_sector_shift, float& out_best, float& out_second) const
{
    out_sector_shift = 0;
    out_best = 2.0f;
    out_second = 2.0f;
    if (m_scan_context_descriptors.empty() || m_scan_context_descriptors.back().size() == 0) {
        return -1;
    }
    const auto& query = m_scan_context_descriptors.back();
    const auto& query_key = m_scan_context_ring_keys.back();
    const double current_time = m_key_poses.back().time;

    // Two-stage retrieval: first rank candidates by ring-key L2 distance
    // (fast coarse filter), then score the top-K via column-shifted cosine
    // distance on the full descriptor.
    std::vector<std::pair<float, int>> ranked;  // (ring-key dist, idx)
    ranked.reserve(m_scan_context_descriptors.size());
    const size_t cur_idx = m_key_poses.size() - 1;
    for (size_t i = 0; i < cur_idx; i++) {
        if (m_scan_context_descriptors[i].size() == 0) continue;
        if (std::abs(current_time - m_key_poses[i].time) <= m_config.loop_time_thresh) {
            continue;  // too recent — not a true loop candidate
        }
        const float key_dist = (m_scan_context_ring_keys[i] - query_key).norm();
        ranked.emplace_back(key_dist, static_cast<int>(i));
    }
    if (ranked.empty()) return -1;

    const int top_k_count = std::min(
        static_cast<int>(ranked.size()), m_scan_context_config.candidate_top_k);
    std::partial_sort(
        ranked.begin(), ranked.begin() + top_k_count, ranked.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });

    float best_dist = std::numeric_limits<float>::max();
    float second_dist = std::numeric_limits<float>::max();
    int best_idx_unfiltered = -1;
    float best_dist_filtered = static_cast<float>(m_scan_context_config.match_threshold);
    int best_idx = -1;
    int best_shift = 0;
    for (int rank = 0; rank < top_k_count; rank++) {
        const int idx = ranked[rank].second;
        const auto [distance, shift] = scan_context::best_distance(
            query, m_scan_context_descriptors[idx]);
        if (distance < best_dist) {
            second_dist = best_dist;  // demote previous best
            best_dist = distance;
            best_idx_unfiltered = idx;
        } else if (distance < second_dist) {
            second_dist = distance;
        }
        if (distance < best_dist_filtered) {
            best_dist_filtered = distance;
            best_idx = idx;
            best_shift = shift;
        }
    }

    // Lowe-style distinctiveness signal: a true revisit is sharply closer than
    // any other place; in self-similar scenes (open grass) every candidate
    // matches about equally -> best ~ second -> ambiguous, reject-worthy.
    out_best = best_dist < 2.0f ? best_dist : 2.0f;
    out_second = second_dist < 2.0f ? second_dist : 2.0f;
    out_sector_shift = best_shift;
    return best_idx;
}

void SimplePGO::searchForLoopPairs()
{
    if (m_key_poses.size() < 10)
        return;
    if (m_config.min_loop_detect_duration > 0.0)
    {
        if (m_history_pairs.size() > 0)
        {
            double current_time = m_key_poses.back().time;
            double last_time = m_key_poses[m_history_pairs.back().second].time;
            if (current_time - last_time < m_config.min_loop_detect_duration)
                return;
        }
    }

    size_t cur_idx = m_key_poses.size() - 1;

    // Feature-poverty gate: a scan with no spatially-spread structure (open
    // grass: returns clustered near the sensor) can't reliably place itself;
    // any closure it proposes is noise. Skip loop search entirely -> PGO
    // becomes a no-op here (the intended behavior on feature-poor recordings).
    // Guarded by use_scan_context (needs descriptors). The observability half
    // of the gate (degeneracy) runs later, once a candidate's source submap
    // exists. min_descriptor_std is retained but defaults off (structure
    // overlaps too much between scenes); occupancy is the effective gate.
    if (m_config.use_scan_context && !m_scan_context_descriptors.empty() &&
        m_scan_context_descriptors.back().size() > 0)
    {
        const auto& descriptor = m_scan_context_descriptors.back();
        if (m_config.min_descriptor_std > 0.0 &&
            scan_context::descriptor_structure(descriptor) < m_config.min_descriptor_std)
            return;
        if (m_config.loop_min_occupancy > 0 &&
            scan_context::descriptor_occupancy(descriptor) < m_config.loop_min_occupancy)
            return;
    }

    int loop_idx = -1;
    int sector_shift = 0;
    float sc_best = 2.0f;
    float sc_second = 2.0f;
    if (m_config.use_scan_context) {
        loop_idx = searchByScanContext(sector_shift, sc_best, sc_second);
    }
    if (loop_idx < 0) {
        // Fallback (or sole path if SC disabled): kdtree on past positions.
        loop_idx = searchByPosition();
    }

    if (loop_idx < 0)
        return;

    // Scan-context can false-match on similar structure (parking lots,
    // intersections) anywhere in the trajectory. Skip if positionally
    // implausible, so the ICP gate isn't the only filter.
    if (m_config.loop_candidate_max_distance_m > 0.0)
    {
        double candidate_distance =
            (m_key_poses[cur_idx].t_global - m_key_poses[loop_idx].t_global).norm();
        if (candidate_distance > m_config.loop_candidate_max_distance_m)
            return;
    }

    // Use Scan Context's column shift to seed ICP with a yaw-aligned initial
    // guess, which dramatically improves convergence on revisits at
    // different headings. Both submaps are in *global* frame, so a naive
    // rotation about the world origin would translate the source cloud
    // kilometers off (e.g. at world position (3500, 350), rotating by 90°
    // sends it to (-350, 3500)). Build a transform that rotates about the
    // source keyframe's own global position instead:
    //     init = T(source_position) · Rz(yaw) · T(-source_position)
    // → init · p = rotation · (p - source_position) + source_position
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    if (m_config.use_scan_context && sector_shift != 0) {
        const double yaw = scan_context::yaw_from_shift(sector_shift, m_scan_context_config.n_sectors);
        Eigen::AngleAxisf yaw_axis_angle(
            static_cast<float>(yaw), Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotation = yaw_axis_angle.toRotationMatrix();
        Eigen::Vector3f source_position = m_key_poses[cur_idx].t_global.cast<float>();
        init_guess.block<3, 3>(0, 0) = rotation;
        init_guess.block<3, 1>(0, 3) = source_position - rotation * source_position;
    }

    CloudType::Ptr target_cloud = getSubMap(loop_idx, m_config.loop_submap_half_range, m_config.submap_resolution);
    CloudType::Ptr source_cloud = getSubMap(cur_idx, 0, m_config.submap_resolution);
    CloudType::Ptr align_cloud(new CloudType);

    m_icp.setInputSource(source_cloud);
    m_icp.setInputTarget(target_cloud);
    m_icp.align(*align_cloud, init_guess);

    // Observability gate: even with a positionally-plausible candidate and low
    // ICP fitness, a planar/degenerate source scan (open grass) leaves the
    // alignment unconstrained in-plane — fitness lies. Reject when the smallest
    // normalized normal-scatter eigenvalue is below threshold.
    float degeneracy_min = -1.0f;
    float degeneracy_mid = -1.0f;
    if (m_config.loop_min_degeneracy > 0.0 || m_config.debug)
        cloud_degeneracy(source_cloud, degeneracy_min, degeneracy_mid);

    if (m_config.debug)
    {
        double cand_dist =
            (m_key_poses[cur_idx].t_global - m_key_poses[loop_idx].t_global).norm();
        const bool have_desc =
            m_config.use_scan_context && !m_scan_context_descriptors.empty();
        float structure = have_desc
            ? scan_context::descriptor_structure(m_scan_context_descriptors.back())
            : -1.0f;
        int occupancy = have_desc
            ? scan_context::descriptor_occupancy(m_scan_context_descriptors.back())
            : -1;
        // Lowe ratio: best / second-best Scan-Context distance. ~1 => ambiguous
        // (every place matches equally, e.g. grass); << 1 => a distinctive revisit.
        float lowe_ratio = (sc_second > 1e-6f) ? sc_best / sc_second : -1.0f;
        bool accepted = m_icp.hasConverged() &&
                        m_icp.getFitnessScore() <= m_config.loop_score_thresh;
        fprintf(stderr,
                "PGO_DIAG kf=%zu cand=%d dist=%.2f fitness=%.5f converged=%d structure=%.2f "
                "occ=%d sc_best=%.3f sc_2nd=%.3f lowe=%.3f degen_min=%.4f degen_mid=%.4f "
                "src_pts=%zu tgt_pts=%zu accepted=%d\n",
                cur_idx, loop_idx, cand_dist, m_icp.getFitnessScore(),
                m_icp.hasConverged() ? 1 : 0, structure, occupancy, sc_best, sc_second,
                lowe_ratio, degeneracy_min, degeneracy_mid,
                source_cloud->size(), target_cloud->size(), accepted ? 1 : 0);
    }

    if (m_config.loop_min_degeneracy > 0.0 && degeneracy_min >= 0.0 &&
        degeneracy_min < m_config.loop_min_degeneracy)
        return;

    if (!m_icp.hasConverged() || m_icp.getFitnessScore() > m_config.loop_score_thresh)
        return;

    M4F loop_transform = m_icp.getFinalTransformation();

    LoopPair one_pair;
    one_pair.source_id = cur_idx;
    one_pair.target_id = loop_idx;
    one_pair.score = m_icp.getFitnessScore();
    M3D r_refined = loop_transform.block<3, 3>(0, 0).cast<double>() * m_key_poses[cur_idx].r_global;
    V3D t_refined = loop_transform.block<3, 3>(0, 0).cast<double>() * m_key_poses[cur_idx].t_global + loop_transform.block<3, 1>(0, 3).cast<double>();
    one_pair.r_offset = m_key_poses[loop_idx].r_global.transpose() * r_refined;
    one_pair.t_offset = m_key_poses[loop_idx].r_global.transpose() * (t_refined - m_key_poses[loop_idx].t_global);
    // Original isotropic noise = ICP fitness on all 6 DOF (unchanged behavior).
    one_pair.noise = gtsam::noiseModel::Diagonal::Variances(
        gtsam::Vector6::Ones() * one_pair.score);
    m_cache_pairs.push_back(one_pair);
    m_history_pairs.emplace_back(one_pair.target_id, one_pair.source_id);
}

void SimplePGO::addLocationConstraintFactors(size_t node_idx, const LocationConstraintObs &constraint)
{
    const KeyPoseWithCloud &node = m_key_poses[node_idx];

    // Ensure a graph variable for this location id (Symbol('l', index)).
    auto found = m_location_index.find(constraint.to_id);
    int loc_idx;
    bool is_new = (found == m_location_index.end());
    if (is_new)
    {
        loc_idx = m_next_location++;
        m_location_index[constraint.to_id] = loc_idx;
    }
    else
    {
        loc_idx = found->second;
        m_location_closure = true;  // re-sighting => a loop closure
    }
    gtsam::Key loc_key = gtsam::Symbol('l', loc_idx);

    if (is_new)
    {
        // Initialize the location in the world frame from this node.
        M3D r_loc_world = node.r_global * constraint.r_body_loc;
        V3D t_loc_world = node.r_global * constraint.t_body_loc + node.t_global;
        m_initial_values.insert(
            loc_key, gtsam::Pose3(gtsam::Rot3(r_loc_world), gtsam::Point3(t_loc_world)));
    }

    // Revision: a constraint reusing an existing constraint_instance_id supersedes
    // the committed factors carrying that id -> schedule them for removal. Lets an
    // external estimator roll out a better estimate (improving tag/GPS lock) and
    // drop its earlier ones.
    if (!constraint.constraint_instance_id.empty())
    {
        auto committed = m_committed_by_instance.find(constraint.constraint_instance_id);
        if (committed != m_committed_by_instance.end() && !committed->second.empty())
        {
            for (size_t factor_index : committed->second)
                m_pending_removals.push_back(factor_index);
            committed->second.clear();
            m_location_closure = true;  // removal also earns the extra relin passes
        }
    }

    // Noise model = the covariance carried in the message (already in GTSAM Pose3
    // tangent order [rot(3), trans(3)]) -> straight into the factor. Robust kernel
    // optional.
    gtsam::SharedNoiseModel noise = gtsam::noiseModel::Gaussian::Covariance(constraint.covariance);
    if (m_config.loop_robust_kernel)
        noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(m_config.loop_robust_huber_k),
            noise);

    // Observation factor: node -> location relative pose = T_body_loc.
    size_t graph_pos = m_graph.size();
    m_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        node_idx, loc_key,
        gtsam::Pose3(gtsam::Rot3(constraint.r_body_loc), gtsam::Point3(constraint.t_body_loc)),
        noise));
    m_staged_constraint_factors.push_back({constraint.constraint_instance_id, graph_pos});

    if (m_config.debug)
        fprintf(stderr,
                "PGO_LOCATION node=%zu to_id=%s new=%d |t_body_loc|=%.2f instance=%s\n",
                node_idx, constraint.to_id.c_str(), is_new ? 1 : 0,
                constraint.t_body_loc.norm(), constraint.constraint_instance_id.c_str());
}

void SimplePGO::smoothAndUpdate()
{
    bool has_loop = !m_cache_pairs.empty();
    // 添加回环因子
    if (has_loop)
    {
        for (LoopPair &pair : m_cache_pairs)
        {
            gtsam::SharedNoiseModel noise = pair.noise;
            // Robust (M-estimator) kernel down-weights an outlier closure
            // instead of letting it bend the whole graph. Off by default.
            if (m_config.loop_robust_kernel)
                noise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Huber::Create(m_config.loop_robust_huber_k),
                    noise);
            m_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(pair.target_id, pair.source_id,
                                                           gtsam::Pose3(gtsam::Rot3(pair.r_offset),
                                                                        gtsam::Point3(pair.t_offset)),
                                                           noise));
        }
        std::vector<LoopPair>().swap(m_cache_pairs);
    }
    // A re-sighted location closes a loop just like a lidar closure, so it
    // earns the same extra relinearization passes (a large correction needs them).
    bool has_closure = has_loop || m_location_closure;

    // smooth and mapping. removeFactorIndices applies constraint revision: factors
    // superseded by a constraint reusing the same constraint_instance_id are dropped.
    gtsam::FactorIndices remove(m_pending_removals.begin(), m_pending_removals.end());
    gtsam::ISAM2Result result = m_isam2->update(m_graph, m_initial_values, remove);
    m_isam2->update();
    if (has_closure)
    {
        m_isam2->update();
        m_isam2->update();
        m_isam2->update();
        m_isam2->update();
    }

    // Record the iSAM2 factor index assigned to each staged constraint factor so a
    // future revision (same constraint_instance_id) can remove it. Constraints with
    // an empty instance id are not tracked (no revision capability).
    for (const StagedConstraintFactor &staged : m_staged_constraint_factors)
    {
        if (!staged.instance_id.empty() && staged.graph_pos < result.newFactorsIndices.size())
            m_committed_by_instance[staged.instance_id].push_back(
                result.newFactorsIndices[staged.graph_pos]);
    }
    m_staged_constraint_factors.clear();
    m_pending_removals.clear();
    m_location_closure = false;

    m_graph.resize(0);
    m_initial_values.clear();

    // update key poses
    gtsam::Values estimate_values = m_isam2->calculateBestEstimate();
    for (size_t i = 0; i < m_key_poses.size(); i++)
    {
        gtsam::Pose3 pose = estimate_values.at<gtsam::Pose3>(i);
        m_key_poses[i].r_global = pose.rotation().matrix().cast<double>();
        m_key_poses[i].t_global = pose.translation().matrix().cast<double>();
    }
    // update offset
    const KeyPoseWithCloud &last_item = m_key_poses.back();
    m_r_offset = last_item.r_global * last_item.r_local.transpose();
    m_t_offset = last_item.t_global - m_r_offset * last_item.t_local;
}
