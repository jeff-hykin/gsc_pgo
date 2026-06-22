// Lockstep unit test for the landmark-event / gravity-anchor / replacement paths
// of SimplePGO. Links the PGO sources directly (no LCM, no main loop) and drives
// keyframes + landmark observations exactly as main.cpp would, then asserts the
// optimized graph behaves as designed. Built as `pgo_landmark_test` and run from
// result/bin. Exits non-zero on any failed assertion.

#include <cmath>
#include <cstdio>
#include <string>

#include "commons.h"
#include "simple_pgo.h"

static int g_failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_failures++; }      \
        else { fprintf(stderr, "ok:   %s\n", msg); }                            \
    } while (0)

// Feed one keyframe (identity orientation unless r given) at world (x,y,z) with
// optional landmark observations, then run a smoothAndUpdate cycle.
static void step(SimplePGO& pgo, double t, const V3D& xyz,
                 const std::vector<LandmarkObs>& obs, const M3D& r = M3D::Identity()) {
    CloudWithPose cwp;
    cwp.cloud = nullptr;
    cwp.pose.r = r;
    cwp.pose.t = xyz;
    cwp.pose.second = t;
    cwp.pose.sec = static_cast<int32_t>(t);
    cwp.pose.nsec = 0;
    cwp.landmark_obs = obs;
    if (pgo.addKeyPose(cwp)) {
        pgo.searchForLoopPairs();
        pgo.smoothAndUpdate();
    }
}

static LandmarkObs lm(const std::string& id, const M3D& r_body_lm,
                      const V3D& t_body_lm, double ts, double replacement_ms = 0.0) {
    LandmarkObs o;
    o.id = id;
    o.r_body_lm = r_body_lm;
    o.t_body_lm = t_body_lm;
    o.ts = ts;
    o.confidence = 1.0;
    o.replacement_ms = replacement_ms;
    return o;
}

static M3D yaw(double a) { return M3D(Eigen::AngleAxisd(a, V3D::UnitZ())); }

// A drifted loop: the robot drives a circle of `radius` over `K` steps and
// physically returns to the start, but the odometry carries a constant per-step
// yaw bias, so the *fed* trajectory spirals open and the end lands far from the
// start. This mirrors the real dataset (drift = accumulated rotation distributed
// over thousands of edges), so a loop-closing landmark factor can redistribute a
// small yaw correction per edge and pull the end back — which a single stiff edge
// could never allow.
struct LoopFrame { V3D true_pos; double true_yaw; V3D odom_pos; double odom_yaw; };

static std::vector<LoopFrame> make_drifted_loop(int K, double radius, double yaw_bias) {
    std::vector<LoopFrame> f(K + 1);
    auto true_pose = [&](int i, V3D& p, double& th) {
        double a = 2.0 * M_PI * i / K;
        p = V3D(radius * std::cos(a), radius * std::sin(a), 0.0);
        th = a + M_PI / 2.0;  // tangent heading
    };
    V3D p0; double th0; true_pose(0, p0, th0);
    f[0] = {p0, th0, p0, th0};
    for (int i = 1; i <= K; i++) {
        V3D pi, pm; double thi, thm;
        true_pose(i, pi, thi);
        true_pose(i - 1, pm, thm);
        V3D dp_body = yaw(thm).transpose() * (pi - pm);  // relative motion, body frame
        double dth = thi - thm;
        double odom_yaw = f[i - 1].odom_yaw + dth + yaw_bias;  // biased turn
        V3D odom_pos = f[i - 1].odom_pos + yaw(f[i - 1].odom_yaw) * dp_body;
        f[i] = {pi, thi, odom_pos, odom_yaw};
    }
    return f;
}

// Observation of a fixed world landmark from a TRUE pose. A short-range landmark
// near the start (like the real 0.5 m AprilTags) makes the loop-closure residual
// reflect the position drift directly (a far/centered landmark would have its
// view rotate with the robot and hide the drift).
static const V3D LMK_WORLD(32.0, 0.0, 0.0);  // ~2 m beyond the start point (30,0)

static LandmarkObs obs_at(const std::string& id, const LoopFrame& fr, double ts,
                          double replacement_ms = 0.0) {
    V3D t = yaw(fr.true_yaw).transpose() * (LMK_WORLD - fr.true_pos);
    return lm(id, M3D::Identity(), t, ts, replacement_ms);
}

// --- Test 1: two sightings of one landmark close a drifted loop, pulling the
// (badly drifted) end back near the true end (the start).
static void test_landmark_closure() {
    Config cfg;
    cfg.use_landmarks = true;
    cfg.key_pose_delta_trans = 0.5;
    SimplePGO pgo(cfg);

    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);
    for (int i = 0; i <= K; i++) {
        std::vector<LandmarkObs> obs;
        if (i == 0 || i == K) obs.push_back(obs_at("C", f[i], i * 0.1));  // two sightings
        step(pgo, i * 0.1, f[i].odom_pos, obs, yaw(f[i].odom_yaw));
    }
    const auto& kps = pgo.keyPoses();
    // Error against the TRUE end pose (the loop returns to its start on the circle).
    double fed_err = (f[K].odom_pos - f[K].true_pos).norm();
    double cor_err = (kps.back().t_global - f[K].true_pos).norm();
    fprintf(stderr,
            "   true end=(%.1f,%.1f) fed end=(%.1f,%.1f) err %.1f m -> "
            "corrected=(%.1f,%.1f) err %.1f m\n",
            f[K].true_pos.x(), f[K].true_pos.y(), f[K].odom_pos.x(), f[K].odom_pos.y(),
            fed_err, kps.back().t_global.x(), kps.back().t_global.y(), cor_err);
    CHECK(fed_err > 20.0, "odometry drift opens the loop by >20 m");
    CHECK(cor_err < fed_err * 0.5,
          "landmark closure pulls the drifted end back to the true end by >50%");
}

// --- Test 2: the gravity anchor preserves the initial roll/pitch. Start with a
// 30 deg pitch; after landmark-driven optimization, keyframe 0's pitch is unchanged.
static double pitch_deg(const M3D& r) { return std::asin(-r(2, 0)) * 180.0 / M_PI; }

static void test_gravity_anchor() {
    Config cfg;
    cfg.use_landmarks = true;
    cfg.key_pose_delta_trans = 0.5;
    SimplePGO pgo(cfg);

    const double pitch0 = 30.0 * M_PI / 180.0;
    M3D r0(Eigen::AngleAxisd(pitch0, V3D::UnitY()));  // 30 deg pitch about y
    const M3D I = M3D::Identity();
    const V3D L_body(-1.0, 0.0, 0.0);

    // A drifted loop (so landmark closures actively pull on the graph) whose
    // first keyframe is pitched 30 deg. The anchor must keep that pitch.
    step(pgo, 0.0, V3D(0, 0, 0), {lm("G", I, L_body, 0.0)}, r0);
    step(pgo, 1.0, V3D(10, 0, 0), {});
    step(pgo, 2.0, V3D(10, 12, 0), {});
    step(pgo, 3.0, V3D(0, 8, 0), {});
    step(pgo, 4.0, V3D(5, 5, 0), {lm("G", I, L_body, 4.0)}, r0);

    double p_in = 30.0;
    double p_out = pitch_deg(pgo.keyPoses().front().r_global);
    fprintf(stderr, "   kf0 pitch in=%.4f out=%.4f deg\n", p_in, p_out);
    CHECK(std::abs(p_out - p_in) < 0.1,
          "gravity anchor keeps kf0 pitch within 0.1 deg of input");
}

// --- Test 3: replacement supersedes a stale (wrong) committed landmark factor.
// kf0 commits a badly-wrong landmark pose; a corrective sighting at kf1 with
// replacement removes it; the closure at the end then uses the correct geometry,
// so the end is pulled back near the origin (a wrong, un-replaced factor would
// drag the solution off).
static void test_replacement() {
    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);

    // Baseline: NO replacement — a badly-wrong first sighting of the landmark at
    // kf0 stays in the graph and fights the correct revisit, so the closure is
    // poisoned (the end does not come back cleanly).
    auto run = [&](bool corrective) {
        Config cfg;
        cfg.use_landmarks = true;
        cfg.key_pose_delta_trans = 0.5;
        SimplePGO pgo(cfg);
        for (int i = 0; i <= K; i++) {
            std::vector<LandmarkObs> obs;
            if (i == 0) {
                // Wrong first estimate (off by 15 m in body-y).
                LandmarkObs bad = obs_at("R", f[0], 0.0);
                bad.t_body_lm += V3D(0, 15, 0);
                obs.push_back(bad);
            } else if (i == 1 && corrective) {
                // Corrective sighting supersedes the stale one (replacement 5 s).
                obs.push_back(obs_at("R", f[1], 0.1, 5000.0));
            } else if (i == K) {
                obs.push_back(obs_at("R", f[K], i * 0.1));
            }
            step(pgo, i * 0.1, f[i].odom_pos, obs, yaw(f[i].odom_yaw));
        }
        return (pgo.keyPoses().back().t_global - f[K].true_pos).norm();  // err vs true end
    };

    double poisoned = run(false);
    double replaced = run(true);
    fprintf(stderr, "   end error vs true: no-replacement=%.1f m, with-replacement=%.1f m\n",
            poisoned, replaced);
    CHECK(replaced < poisoned - 5.0,
          "replacement removes the stale wrong landmark -> cleaner closure than without");
}

// --- Test 4: the per-keyframe gravity anchor keeps EVERY keyframe level through
// a loop closure (not just kf0). The keyframes drive on flat ground (zero
// roll/pitch); after the landmark closure corrects the drift, no inner keyframe
// should have tilted — tilt is what converts horizontal travel into vertical and
// corrupts z. Compares anchor off vs on.
static void test_per_keyframe_gravity() {
    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);
    auto run = [&](bool per_kf) {
        Config cfg;
        cfg.use_landmarks = true;
        cfg.key_pose_delta_trans = 0.5;
        cfg.gravity_anchor_per_keyframe = per_kf;
        SimplePGO pgo(cfg);
        for (int i = 0; i <= K; i++) {
            std::vector<LandmarkObs> obs;
            if (i == 0 || i == K) obs.push_back(obs_at("P", f[i], i * 0.1));
            step(pgo, i * 0.1, f[i].odom_pos, obs, yaw(f[i].odom_yaw));
        }
        double max_pitch = 0.0;
        for (const auto& kp : pgo.keyPoses())
            max_pitch = std::max(max_pitch, std::abs(pitch_deg(kp.r_global)));
        return max_pitch;
    };
    double off = run(false), on = run(true);
    fprintf(stderr, "   max |pitch| across keyframes: anchor off=%.2f deg, on=%.2f deg\n", off, on);
    CHECK(on < 0.5,
          "per-keyframe gravity anchor keeps all keyframes level (<0.5 deg) through closure");
}

int main() {
    fprintf(stderr, "== test_landmark_closure ==\n");
    test_landmark_closure();
    fprintf(stderr, "== test_gravity_anchor ==\n");
    test_gravity_anchor();
    fprintf(stderr, "== test_per_keyframe_gravity ==\n");
    test_per_keyframe_gravity();
    fprintf(stderr, "== test_replacement ==\n");
    test_replacement();
    if (g_failures) {
        fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
        return 1;
    }
    fprintf(stderr, "\nall assertions passed\n");
    return 0;
}
