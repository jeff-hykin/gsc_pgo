// PGO NativeModule — faithful port of pgo_node.cpp from ROS2 to LCM.
// Subscribes to registered_scan + odometry, runs SimplePGO (iSAM2 + PCL ICP),
// publishes corrected_odometry, _global_map (debug, off by default), and TF correction offset.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <queue>
#include <set>
#include <signal.h>
#include <thread>
#include <vector>

#include <lcm/lcm-cpp.hpp>
#include <Eigen/Geometry>
#include <pcl/console/print.h>

#include "commons.h"
#include "simple_pgo.h"
#include "dimos_native_module.hpp"
#include "msgs/Graph3D.hpp"
#include "msgs/GraphDelta3D.hpp"
#include "msgs/Landmark.hpp"
#include "point_cloud_utils.hpp"

#include "nav_msgs/Odometry.hpp"
#include "nav_msgs/Path.hpp"
#include "sensor_msgs/PointCloud2.hpp"
#include "geometry_msgs/Pose.hpp"
#include "geometry_msgs/PoseStamped.hpp"
#include "geometry_msgs/Quaternion.hpp"
#include "geometry_msgs/Point.hpp"
#include "geometry_msgs/Transform.hpp"
#include "geometry_msgs/TransformStamped.hpp"
#include "tf2_msgs/TFMessage.hpp"

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false); }

// Shared state between LCM callbacks and main loop
static std::mutex g_buffer_mutex;
static std::queue<CloudWithPose> g_cloud_buffer;
static double g_last_message_time = 0.0;
// Bounded FIFO depth: keep at most this many pending scans, dropping the
// oldest when full. <=0 means unbounded. Big enough that an ack-gated eval
// replay (~1 scan in flight) never drops; small enough that a live PGO that
// falls behind has bounded latency/memory instead of an unbounded backlog.
static std::atomic<int> g_max_scan_queue{0};

// Latest odometry for non-keyframe TF broadcasting
static std::mutex g_odom_mutex;
static M3D g_latest_r = M3D::Identity();
static V3D g_latest_t = V3D::Zero();
static double g_latest_time = 0.0;
static bool g_has_odom = false;

// --- Static-frame resolution --------------------------------------------------
// A Landmark event's pose is in its own frame_id (typically a camera frame). We
// resolve that frame to the robot body via the *static* TF tree (/tf_static) —
// the fixed camera mount. Frames that only exist on the dynamic /tf tree (map,
// odom) won't resolve, which is exactly right: their pose already has odometry
// drift baked in and is useless for correcting drift.
struct Tf { M3D r = M3D::Identity(); V3D t = V3D::Zero(); };  // p_out = r*p_in + t
static Tf tf_compose(const Tf& a, const Tf& b) { return {a.r * b.r, a.r * b.t + a.t}; }
static Tf tf_inverse(const Tf& a) { M3D ri = a.r.transpose(); return {ri, -ri * a.t}; }

// --- Landmark-event state ------------------------------------------------------
// Decoupled Landmark events (a separate perceiver emits them). A Landmark's pose
// is the landmark in its header frame_id (a camera frame); we resolve that frame
// to the body via /tf_static, compose landmark-in-body, and buffer it for
// time-association to a keyframe. The detection + noise/confidence filtering
// happen upstream in the perceiver.
static std::mutex g_landmark_mutex;
static std::deque<LandmarkObs> g_landmark_buffer;
static bool g_use_landmarks = false;
static double g_landmark_assoc_max_dt = 0.2;    // s: max |lm_ts - keyframe_ts|
static double g_landmark_buffer_window = 3.0;   // s: how long to retain landmarks

// Static TF tree: frame -> [(neighbor, T_neighbor_frame)], built from /tf_static.
static std::mutex g_tf_mutex;
static std::map<std::string, std::vector<std::pair<std::string, Tf>>> g_static_tf;
static std::string g_body_frame = "base_link";

// Resolve T_target_source (p_target = T * p_source) over the static tree. BFS;
// returns false if no static path exists (e.g. source is a dynamic map/odom
// frame, or no /tf_static received yet).
static bool resolve_static_tf(const std::string& target, const std::string& source, Tf& out) {
    if (source == target) { out = Tf{}; return true; }
    std::lock_guard<std::mutex> lock(g_tf_mutex);
    std::map<std::string, Tf> dist;
    dist[source] = Tf{};
    std::queue<std::string> q;
    q.push(source);
    while (!q.empty()) {
        std::string n = q.front();
        q.pop();
        auto it = g_static_tf.find(n);
        if (it == g_static_tf.end()) continue;
        for (const auto& [m, t_m_n] : it->second) {
            if (dist.count(m)) continue;
            dist[m] = tf_compose(t_m_n, dist[n]);  // T_m_source = T_m_n . T_n_source
            if (m == target) { out = dist[m]; return true; }
            q.push(m);
        }
    }
    return false;
}

class Handlers {
public:
    void on_odometry(const lcm::ReceiveBuffer*, const std::string&,
                     const nav_msgs::Odometry* msg) {
        M3D r = Eigen::Quaterniond(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        ).toRotationMatrix();

        V3D t(msg->pose.pose.position.x,
               msg->pose.pose.position.y,
               msg->pose.pose.position.z);

        double ts = msg->header.stamp.sec + msg->header.stamp.nsec / 1e9;

        std::lock_guard<std::mutex> lock(g_odom_mutex);
        g_latest_r = r;
        g_latest_t = t;
        g_latest_time = ts;
        g_has_odom = true;
    }

    void on_registered_scan(const lcm::ReceiveBuffer*, const std::string&,
                            const sensor_msgs::PointCloud2* msg) {
        std::lock_guard<std::mutex> odom_lock(g_odom_mutex);
        if (!g_has_odom)
            return;

        double ts = g_latest_time;

        // Reject out-of-order messages
        if (ts < g_last_message_time)
            return;
        g_last_message_time = ts;

        CloudWithPose cloud_with_pose;
        cloud_with_pose.pose.r = g_latest_r;
        cloud_with_pose.pose.t = g_latest_t;
        cloud_with_pose.pose.setTime(static_cast<int32_t>(ts),
                        static_cast<uint32_t>((ts - static_cast<int32_t>(ts)) * 1e9));

        // Parse PointCloud2 to PCL
        cloud_with_pose.cloud = CloudType::Ptr(new CloudType);
        smartnav::to_pcl(*msg, *cloud_with_pose.cloud);

        std::lock_guard<std::mutex> buf_lock(g_buffer_mutex);
        g_cloud_buffer.push(cloud_with_pose);
        int cap = g_max_scan_queue.load();
        while (cap > 0 && static_cast<int>(g_cloud_buffer.size()) > cap) {
            g_cloud_buffer.pop();  // drop oldest stale scan to bound the backlog
        }
    }

    // Accumulate the static TF tree from /tf_static.
    void on_static_tf(const lcm::ReceiveBuffer*, const std::string&,
                      const tf2_msgs::TFMessage* msg) {
        std::lock_guard<std::mutex> lock(g_tf_mutex);
        for (const auto& ts : msg->transforms) {
            const std::string& parent = ts.header.frame_id;
            const std::string& child = ts.child_frame_id;
            Tf t_parent_child;
            t_parent_child.r = Eigen::Quaterniond(
                ts.transform.rotation.w, ts.transform.rotation.x,
                ts.transform.rotation.y, ts.transform.rotation.z).toRotationMatrix();
            t_parent_child.t = V3D(ts.transform.translation.x,
                                   ts.transform.translation.y,
                                   ts.transform.translation.z);
            g_static_tf[child].push_back({parent, t_parent_child});
            g_static_tf[parent].push_back({child, tf_inverse(t_parent_child)});
        }
    }

    // A decoupled Landmark event. Its pose is the landmark in header.frame_id (a
    // camera frame); resolve that to the body via /tf_static, compose
    // landmark-in-body, and buffer it for time-association to a keyframe.
    void on_landmark(const lcm::ReceiveBuffer*, const std::string&,
                     const jnav::Landmark* msg) {
        if (!g_use_landmarks)
            return;
        const std::string& frame_id =
            !msg->frame_id.empty() ? msg->frame_id : g_body_frame;
        Tf t_body_frame;
        if (!resolve_static_tf(g_body_frame, frame_id, t_body_frame)) {
            if (g_tf_warned.insert(frame_id).second)
                fprintf(stderr,
                        "PGO: landmark frame '%s' has no static path to '%s' "
                        "(dynamic/map frame?); skipping its landmarks\n",
                        frame_id.c_str(), g_body_frame.c_str());
            return;
        }
        Tf t_frame_lm;
        t_frame_lm.r = Eigen::Quaterniond(
            msg->quat_w, msg->quat_x, msg->quat_y, msg->quat_z).toRotationMatrix();
        t_frame_lm.t = V3D(msg->pos_x, msg->pos_y, msg->pos_z);
        Tf t_body_lm = tf_compose(t_body_frame, t_frame_lm);

        LandmarkObs obs;
        obs.id = msg->id;
        obs.r_body_lm = t_body_lm.r;
        obs.t_body_lm = t_body_lm.t;
        obs.ts = msg->ts;
        obs.confidence = msg->confidence;
        obs.replacement_ms = msg->replacement;

        std::lock_guard<std::mutex> lock(g_landmark_mutex);
        g_landmark_buffer.push_back(std::move(obs));
        while (!g_landmark_buffer.empty() &&
               msg->ts - g_landmark_buffer.front().ts > g_landmark_buffer_window) {
            g_landmark_buffer.pop_front();
        }
    }

private:
    std::set<std::string> g_tf_warned;  // frames already warned about (once each)
};

// Landmark observations to associate with a keyframe: for each landmark id, the
// buffered sighting closest to the keyframe time within g_landmark_assoc_max_dt.
// This does not consume the buffer (the buffer window expires stale entries);
// addKeyPose only stores these if the scan becomes a keyframe.
static std::vector<LandmarkObs> landmarks_near(double keyframe_ts) {
    std::lock_guard<std::mutex> lock(g_landmark_mutex);
    std::map<std::string, size_t> best_idx;
    std::map<std::string, double> best_dt;
    for (size_t i = 0; i < g_landmark_buffer.size(); i++) {
        double dt = std::abs(g_landmark_buffer[i].ts - keyframe_ts);
        if (dt > g_landmark_assoc_max_dt)
            continue;
        auto it = best_dt.find(g_landmark_buffer[i].id);
        if (it == best_dt.end() || dt < it->second) {
            best_dt[g_landmark_buffer[i].id] = dt;
            best_idx[g_landmark_buffer[i].id] = i;
        }
    }
    std::vector<LandmarkObs> out;
    for (const auto& [id, idx] : best_idx)
        out.push_back(g_landmark_buffer[idx]);
    return out;
}

static geometry_msgs::TransformStamped build_tf(const M3D& r, const V3D& t, double ts,
                                                  const std::string& frame_id,
                                                  const std::string& child_frame_id) {
    geometry_msgs::TransformStamped ts_msg;
    ts_msg.header = dimos::make_header(frame_id, ts);
    ts_msg.child_frame_id = child_frame_id;
    Eigen::Quaterniond q(r);
    ts_msg.transform.translation.x = t.x();
    ts_msg.transform.translation.y = t.y();
    ts_msg.transform.translation.z = t.z();
    ts_msg.transform.rotation.x = q.x();
    ts_msg.transform.rotation.y = q.y();
    ts_msg.transform.rotation.z = q.z();
    ts_msg.transform.rotation.w = q.w();
    return ts_msg;
}

static tf2_msgs::TFMessage build_tf_message(const M3D& correction_r,
                                              const V3D& correction_t,
                                              double ts,
                                              const std::string& frame_id,
                                              const std::string& child_frame_id) {
    tf2_msgs::TFMessage msg;
    msg.transforms.push_back(
        build_tf(correction_r, correction_t, ts, frame_id, child_frame_id));
    msg.transforms_length = static_cast<int32_t>(msg.transforms.size());
    return msg;
}

static nav_msgs::Odometry build_odometry(const M3D& r, const V3D& t, double ts,
                                          const std::string& frame_id,
                                          const std::string& child_frame_id) {
    nav_msgs::Odometry odom;
    odom.header = dimos::make_header(frame_id, ts);
    odom.child_frame_id = child_frame_id;

    Eigen::Quaterniond q(r);
    odom.pose.pose.position.x = t.x();
    odom.pose.pose.position.y = t.y();
    odom.pose.pose.position.z = t.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    return odom;
}

// Pose-graph snapshot encoded as a Graph3D:
//   - one node per keyframe
static constexpr uint64_t NODE_KEYFRAME = 0;
static constexpr uint64_t EDGE_ODOMETRY = 0;
static constexpr uint64_t EDGE_LOOP_CLOSURE = 1;

static dimos::Graph3D build_pose_graph(
    const std::vector<KeyPoseWithCloud>& key_poses,
    const std::vector<std::pair<size_t, size_t>>& loop_pairs,
    double ts,
    const std::string& frame_id) {
    dimos::Graph3D msg(frame_id, ts);
    msg.reserve_nodes(key_poses.size());
    msg.reserve_edges(key_poses.size() + loop_pairs.size());
    for (size_t i = 0; i < key_poses.size(); i++) {
        const auto& kp = key_poses[i];
        Eigen::Quaterniond q(kp.r_global);
        msg.add_node(
            static_cast<uint64_t>(i),
            NODE_KEYFRAME,
            kp.time,
            kp.t_global.x(), kp.t_global.y(), kp.t_global.z(),
            q.x(), q.y(), q.z(), q.w());
    }
    for (size_t i = 1; i < key_poses.size(); i++) {
        msg.add_edge(
            static_cast<uint64_t>(i - 1),
            static_cast<uint64_t>(i),
            key_poses[i].time,
            EDGE_ODOMETRY);
    }
    for (const auto& pair : loop_pairs) {
        if (pair.first >= key_poses.size() || pair.second >= key_poses.size()) continue;
        msg.add_edge(
            static_cast<uint64_t>(pair.first),
            static_cast<uint64_t>(pair.second),
            ts,
            EDGE_LOOP_CLOSURE);
    }
    return msg;
}

// Build a GraphDelta3D from paired pre/post keyframe lists. Each
// (node, transform) pair has:
//   - node    = the keyframe BEFORE iSAM2's smoothAndUpdate, with id =
//               keyframe index and metadata_id = NODE_KEYFRAME.
//   - transform = SE(3) delta such that post = transform * pre.
// Convention matches Python's GraphDelta3D.lcm_decode.
static constexpr uint64_t NODE_KEYFRAME_DELTA = 0;

static dimos::GraphDelta3D build_loop_closure_event(
    const std::vector<std::pair<M3D, V3D>>& pre_poses,
    const std::vector<KeyPoseWithCloud>& post_poses,
    double ts,
    const std::string& frame_id) {
    dimos::GraphDelta3D msg(frame_id, ts);
    size_t count = std::min(pre_poses.size(), post_poses.size());
    msg.reserve(count);
    for (size_t i = 0; i < count; i++) {
        const M3D& pre_r = pre_poses[i].first;
        const V3D& pre_t = pre_poses[i].second;
        const M3D& post_r = post_poses[i].r_global;
        const V3D& post_t = post_poses[i].t_global;

        // SE(3) delta such that post = delta * pre.
        M3D r_delta = post_r * pre_r.transpose();
        V3D t_delta = post_t - r_delta * pre_t;
        Eigen::Quaterniond q_pre(pre_r);
        Eigen::Quaterniond q_delta(r_delta);

        msg.add(
            /* id */ static_cast<uint64_t>(i),
            /* metadata_id */ NODE_KEYFRAME_DELTA,
            /* pose_ts */ post_poses[i].time,
            /* pos_x,y,z */ pre_t.x(), pre_t.y(), pre_t.z(),
            /* quat_x,y,z,w */ q_pre.x(), q_pre.y(), q_pre.z(), q_pre.w(),
            /* translation_x,y,z */ t_delta.x(), t_delta.y(), t_delta.z(),
            /* rotation_x,y,z,w */ q_delta.x(), q_delta.y(), q_delta.z(), q_delta.w());
    }
    return msg;
}

int main(int argc, char** argv)
{
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    dimos::NativeModule native_module(argc, argv);

    // Port topics
    // Stream named "lidar" to match the jnav LoopClosure spec. The scan is
    // paired with the latest odometry pose internally (CloudWithPose), so a raw
    // sensor-frame lidar + odometry is what's expected here.
    std::string scan_topic = native_module.topic("lidar");
    std::string odom_topic = native_module.topic("odometry");
    std::string corrected_odom_topic = native_module.topic("corrected_odometry");
    std::string correction_topic = native_module.topic("correction");
    std::string global_map_topic = native_module.topic("_global_map");
    std::string pose_graph_topic = native_module.topic("pose_graph");
    std::string loop_closure_event_topic = native_module.topic("loop_closure_event");

    // Config parameters
    Config config;
    config.key_pose_delta_deg = native_module.arg_float("key_pose_delta_deg", 10.0f);
    config.key_pose_delta_trans = native_module.arg_float("key_pose_delta_trans", 0.5f);
    config.loop_search_radius = native_module.arg_float("loop_search_radius", 1.0f);
    config.loop_time_thresh = native_module.arg_float("loop_time_thresh", 60.0f);
    config.loop_score_thresh = native_module.arg_float("loop_score_thresh", 0.15f);
    config.loop_submap_half_range = native_module.arg_int("loop_submap_half_range", 5);
    config.submap_resolution = native_module.arg_float("submap_resolution", 0.1f);
    config.min_loop_detect_duration = native_module.arg_float("min_loop_detect_duration", 5.0f);
    config.use_scan_context = native_module.arg_bool("use_scan_context", true);
    config.scan_context_num_rings = native_module.arg_int("scan_context_num_rings", 20);
    config.scan_context_num_sectors = native_module.arg_int("scan_context_num_sectors", 60);
    config.scan_context_max_range_m = native_module.arg_float("scan_context_max_range_m", 80.0f);
    config.scan_context_top_k = native_module.arg_int("scan_context_top_k", 10);
    config.scan_context_match_threshold = native_module.arg_float("scan_context_match_threshold", 0.4f);
    config.scan_context_lidar_height_m = native_module.arg_float("scan_context_lidar_height_m", 2.0f);
    config.loop_candidate_max_distance_m = native_module.arg_float("loop_candidate_max_distance_m", 30.0f);
    config.min_descriptor_std = native_module.arg_float("min_descriptor_std", 0.0f);
    config.loop_min_occupancy = native_module.arg_int("loop_min_occupancy", 80);
    config.loop_min_degeneracy = native_module.arg_float("loop_min_degeneracy", 0.05f);

    // Robust (M-estimator) kernel wrapping all loop factors (lidar + landmark).
    config.loop_robust_kernel = native_module.arg_bool("loop_robust_kernel", false);
    config.loop_robust_huber_k = native_module.arg_float("loop_robust_huber_k", 1.345f);

    // Landmark events (decoupled perceiver -> PGO factor-graph manager).
    config.use_landmarks = native_module.arg_bool("use_landmarks", false);
    config.landmark_var_inplane_trans_m2 = native_module.arg_float("landmark_var_inplane_trans_m2", 0.0025f);
    config.landmark_var_range_trans_m2 = native_module.arg_float("landmark_var_range_trans_m2", 0.25f);
    config.landmark_var_yaw_rot_rad2 = native_module.arg_float("landmark_var_yaw_rot_rad2", 0.0025f);
    config.landmark_var_outplane_rot_rad2 = native_module.arg_float("landmark_var_outplane_rot_rad2", 0.04f);
    g_use_landmarks = config.use_landmarks;
    g_landmark_assoc_max_dt = native_module.arg_float("landmark_assoc_max_dt", 0.2f);
    g_landmark_buffer_window = native_module.arg_float("landmark_buffer_window", 3.0f);

    // Gravity anchor (fix the gravity-aligned initial roll/pitch on keyframe 0).
    config.gravity_anchor = native_module.arg_bool("gravity_anchor", true);
    config.gravity_anchor_rp_var = native_module.arg_float("gravity_anchor_rp_var", 1e-12f);
    config.gravity_anchor_yaw_var = native_module.arg_float("gravity_anchor_yaw_var", 1e-12f);
    config.gravity_anchor_trans_var = native_module.arg_float("gravity_anchor_trans_var", 1e-12f);
    config.gravity_anchor_per_keyframe = native_module.arg_bool("gravity_anchor_per_keyframe", false);
    config.gravity_anchor_kf_rp_var = native_module.arg_float("gravity_anchor_kf_rp_var", 1e-4f);

    // Anisotropic odometry between-factor (stiff roll/pitch, looser yaw) — see
    // simple_pgo.h. Preserves z by keeping loop closures from tilting the graph.
    config.odom_rot_rp_var = native_module.arg_float("odom_rot_rp_var", 1e-8f);
    config.odom_rot_yaw_var = native_module.arg_float("odom_rot_yaw_var", 1e-5f);
    config.odom_trans_xy_var = native_module.arg_float("odom_trans_xy_var", 1e-4f);
    config.odom_trans_z_var = native_module.arg_float("odom_trans_z_var", 1e-6f);

    // Node-level config
    std::string frame_id = native_module.arg("frame_id", "map");
    std::string child_frame_id = native_module.arg("child_frame_id", "odom");
    std::string body_frame = native_module.arg("body_frame", "base_link");
    g_body_frame = body_frame;  // target frame for landmark-frame TF resolution
    float global_map_voxel_size = native_module.arg_float("global_map_voxel_size", 0.1f);
    float global_map_publish_rate = native_module.arg_float("global_map_publish_rate", 0.0f);
    // OFF by default (rate <= 0 disables it). Published on the internal `_global_map`
    // port so it never autoconnects to a consumer's `global_map` In — terrain_mapper
    // is the planner's single authoritative global_map. It's a big cloud that can
    // overflow LCM's single-message limit and congest the bus; enable only for debug.
    bool publish_global_map = global_map_publish_rate > 0;
    double global_map_interval = publish_global_map ? 1.0 / global_map_publish_rate : 0.0;

    // Unregister mode: transform world-frame scans to body-frame
    bool unregister_input = native_module.arg_bool("unregister_input", true);

    // Bounded FIFO depth (see g_max_scan_queue). The default is generous enough
    // that an ack-gated eval replay never drops a scan, yet bounds live latency.
    g_max_scan_queue.store(native_module.arg_int("max_scan_queue", 100));

    bool debug = native_module.arg_bool("debug", false);
    config.debug = debug;  // enables PGO_DIAG per-candidate loop-quality logging

    pcl::console::setVerbosityLevel(
        debug ? pcl::console::L_INFO : pcl::console::L_ERROR);

    SimplePGO pgo(config);

    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "PGO: LCM init failed\n");
        return 1;
    }

    Handlers handlers;
    lcm.subscribe(odom_topic, &Handlers::on_odometry, &handlers);
    lcm.subscribe(scan_topic, &Handlers::on_registered_scan, &handlers);
    if (g_use_landmarks && native_module.has("landmarks")) {
        lcm.subscribe(native_module.topic("landmarks"),
                      &Handlers::on_landmark, &handlers);
    }
    // The fixed camera mount (landmark frame -> body) lives on the static tree;
    // subscribe once if landmark ingestion needs it.
    if (g_use_landmarks) {
        std::string tf_static_channel =
            native_module.arg("tf_static_channel", "/tf_static#tf2_msgs.TFMessage");
        lcm.subscribe(tf_static_channel, &Handlers::on_static_tf, &handlers);
    }

    // NativeModule.start() in Python reads stderr for this marker and only
    // returns once it sees it. Without this, upstream publishers can race
    // ahead and emit messages before our LCM subscriptions are live.
    fprintf(stderr, "[DIMOS_NATIVE_READY]\n");
    fflush(stderr);

    if (debug) {
        fprintf(stderr, "PGO native module started\n");
        fprintf(stderr, "  lidar: %s\n", scan_topic.c_str());
        fprintf(stderr, "  odometry: %s\n", odom_topic.c_str());
        fprintf(stderr, "  corrected_odometry: %s\n", corrected_odom_topic.c_str());
        fprintf(stderr, "  correction: %s\n", correction_topic.c_str());
        fprintf(stderr, "  _global_map: %s\n", global_map_topic.c_str());
        fprintf(stderr, "  pose_graph: %s\n", pose_graph_topic.c_str());
        fprintf(stderr, "  loop_closure_event: %s\n", loop_closure_event_topic.c_str());
    }

    double last_global_map_time = 0.0;
    int timer_period_ms = 50;  // 20 Hz, matching original

    while (g_running.load()) {
        // Drain all pending LCM messages
        while (lcm.handleTimeout(0) > 0) {}

        // Strict FIFO: process one scan per tick. The backlog is bounded at
        // enqueue time by g_max_scan_queue (oldest dropped when full).
        CloudWithPose cloud_with_pose;
        bool has_data = false;
        {
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            if (!g_cloud_buffer.empty()) {
                cloud_with_pose = g_cloud_buffer.front();
                g_cloud_buffer.pop();
                has_data = true;
            }
        }

        if (!has_data) {
            std::this_thread::sleep_for(std::chrono::milliseconds(timer_period_ms));
            continue;
        }

        // Optionally transform world-frame scan to body-frame
        if (unregister_input && cloud_with_pose.cloud && cloud_with_pose.cloud->size() > 0) {
            CloudType::Ptr body_cloud(new CloudType);
            // body = R_odom^T * (world_pts - t_odom)
            M3D r_inv = cloud_with_pose.pose.r.transpose();
            for (const auto& pt : *cloud_with_pose.cloud) {
                V3D world_pt(pt.x, pt.y, pt.z);
                V3D body_pt = r_inv * (world_pt - cloud_with_pose.pose.t);
                PointType bp;
                bp.x = static_cast<float>(body_pt.x());
                bp.y = static_cast<float>(body_pt.y());
                bp.z = static_cast<float>(body_pt.z());
                bp.intensity = pt.intensity;
                body_cloud->push_back(bp);
            }
            cloud_with_pose.cloud = body_cloud;
        }

        double cur_time = cloud_with_pose.pose.second;

        // Attach the landmark events nearest this scan's time (no-op if off or
        // none in the association window). addKeyPose stores them only if this
        // scan becomes a keyframe.
        if (g_use_landmarks)
            cloud_with_pose.landmark_obs = landmarks_near(cur_time);

        if (!pgo.addKeyPose(cloud_with_pose)) {
            // Not a keyframe — still broadcast TF and corrected odom
            M3D corr_r = pgo.offsetR() * cloud_with_pose.pose.r;
            V3D corr_t = pgo.offsetR() * cloud_with_pose.pose.t + pgo.offsetT();
            nav_msgs::Odometry corrected = build_odometry(
                corr_r, corr_t, cur_time, frame_id, body_frame);
            lcm.publish(corrected_odom_topic, &corrected);

            auto tf_msg = build_tf_message(
                pgo.offsetR(), pgo.offsetT(), cur_time, frame_id, child_frame_id);
            lcm.publish(correction_topic, &tf_msg);

            std::this_thread::sleep_for(std::chrono::milliseconds(timer_period_ms));
            continue;
        }

        // Keyframe added. Snapshot keyframe global poses BEFORE search +
        // smooth so we can publish the delta applied by iSAM2 if a loop
        // closure actually fires.
        pgo.searchForLoopPairs();
        bool had_loop = pgo.hasLoop();

        std::vector<std::pair<M3D, V3D>> pre_poses;
        if (had_loop) {
            pre_poses.reserve(pgo.keyPoses().size());
            for (const auto& kp : pgo.keyPoses()) {
                pre_poses.emplace_back(kp.r_global, kp.t_global);
            }
        }

        pgo.smoothAndUpdate();

        if (had_loop) {
            dimos::GraphDelta3D loop_closure_event_msg = build_loop_closure_event(
                pre_poses, pgo.keyPoses(), cur_time, frame_id);
            loop_closure_event_msg.publish(lcm, loop_closure_event_topic);
            if (debug) {
                fprintf(stderr,
                        "PGO: loop_closure_event published — %zu keyframe deltas\n",
                        pre_poses.size());
            }
        }

        if (debug) {
            fprintf(stderr, "PGO: keyframe %zu at (%.1f, %.1f, %.1f)\n",
                    pgo.keyPoses().size(),
                    cloud_with_pose.pose.t.x(), cloud_with_pose.pose.t.y(), cloud_with_pose.pose.t.z());
        }

        // Publish corrected odometry
        M3D corr_r = pgo.offsetR() * cloud_with_pose.pose.r;
        V3D corr_t = pgo.offsetR() * cloud_with_pose.pose.t + pgo.offsetT();
        nav_msgs::Odometry corrected = build_odometry(
            corr_r, corr_t, cur_time, frame_id, body_frame);
        lcm.publish(corrected_odom_topic, &corrected);

        auto tf_msg = build_tf_message(
            pgo.offsetR(), pgo.offsetT(), cur_time, frame_id, child_frame_id);
        lcm.publish(correction_topic, &tf_msg);

        // Publish pose graph (on every keyframe — iSAM2 may have
        // re-optimized prior poses on loop closure).
        dimos::Graph3D pose_graph_msg = build_pose_graph(
            pgo.keyPoses(), pgo.historyPairs(), cur_time, frame_id);
        pose_graph_msg.publish(lcm, pose_graph_topic);

        // Publish global map (throttled)
        double now = cur_time;
        if (publish_global_map && now - last_global_map_time >= global_map_interval) {
            last_global_map_time = now;

            if (!pgo.keyPoses().empty()) {
                CloudType::Ptr global_cloud(new CloudType);
                for (size_t i = 0; i < pgo.keyPoses().size(); i++) {
                    CloudType::Ptr world_cloud(new CloudType);
                    pcl::transformPointCloud(
                        *pgo.keyPoses()[i].body_cloud,
                        *world_cloud,
                        pgo.keyPoses()[i].t_global,
                        Eigen::Quaterniond(pgo.keyPoses()[i].r_global));
                    *global_cloud += *world_cloud;
                }

                // Voxel downsample
                CloudType::Ptr filtered(new CloudType);
                pcl::VoxelGrid<PointType> voxel;
                voxel.setInputCloud(global_cloud);
                voxel.setLeafSize(global_map_voxel_size, global_map_voxel_size, global_map_voxel_size);
                voxel.filter(*filtered);

                sensor_msgs::PointCloud2 map_msg = smartnav::from_pcl(*filtered, frame_id, now);
                lcm.publish(global_map_topic, &map_msg);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(timer_period_ms));
    }

    if (debug) fprintf(stderr, "PGO native module shutting down\n");
    return 0;
}
