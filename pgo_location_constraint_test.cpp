// Lockstep unit test for the location-constraint / gravity-anchor / revision
// paths of SimplePGO. Links the PGO sources directly (no LCM, no main loop) and
// drives keyframes + location constraints as main.cpp would, then asserts the
// optimized graph behaves as designed. Built as `pgo_location_constraint_test`
// and run from result/bin. Exits non-zero on any failed assertion.

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

static M3D yaw(double a) { return M3D(Eigen::AngleAxisd(a, V3D::UnitZ())); }

static void set_time(PoseWithTime& p, double t) {
    p.setTime(static_cast<int32_t>(t),
              static_cast<uint32_t>((t - static_cast<int32_t>(t)) * 1e9));
}

// Feed one scan keyframe (identity orientation unless r given) at world (x,y,z),
// then run a smoothAndUpdate cycle. Clouds are null (scan-context disabled), so
// only the location constraints close loops — exactly the path under test.
static void step(SimplePGO& pgo, double t, const V3D& xyz, const M3D& r = M3D::Identity()) {
    CloudWithPose cwp;
    cwp.cloud = nullptr;
    cwp.frame_id = "odom";
    cwp.pose.r = r;
    cwp.pose.t = xyz;
    set_time(cwp.pose, t);
    if (pgo.addKeyPose(cwp)) {
        pgo.searchForLoopPairs();
        pgo.smoothAndUpdate();
    }
}

// A diagonal 6x6 covariance (GTSAM tangent order [rot(3), trans(3)]).
static M6D make_cov(double rot_var, double trans_var) {
    M6D cov = M6D::Zero();
    for (int i = 0; i < 3; i++) cov(i, i) = rot_var;
    for (int i = 3; i < 6; i++) cov(i, i) = trans_var;
    return cov;
}

// Ingest a location constraint: the caller-supplied (odom_pos, odom_r) is the
// interpolated odometry pose at the constraint's time (main.cpp does the
// interpolation; here the test supplies it directly).
static void sight(SimplePGO& pgo, const V3D& odom_pos, const M3D& odom_r, double ts,
                  const LocationConstraintObs& c) {
    PoseWithTime p;
    p.r = odom_r;
    p.t = odom_pos;
    set_time(p, ts);
    if (pgo.addLocationConstraint(p, "base_link", c))
        pgo.smoothAndUpdate();
}

// A drifted loop: the robot drives a circle of `radius` over `K` steps and
// physically returns to the start, but the odometry carries a constant per-step
// yaw bias, so the *fed* trajectory spirals open and the end lands far from the
// start. This mirrors the real dataset (drift = accumulated rotation distributed
// over thousands of edges), so a loop-closing location factor can redistribute a
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

// A fixed world location, observed from a TRUE pose. A short-range location near
// the start makes the loop-closure residual reflect the position drift directly
// (a far/centered location would have its view rotate with the robot and hide it).
static const V3D LOC_WORLD(32.0, 0.0, 0.0);  // ~2 m beyond the start point (30,0)

static LocationConstraintObs constraint_at(const std::string& to_id, const std::string& instance,
                                           const LoopFrame& fr, double ts) {
    LocationConstraintObs c;
    c.to_id = to_id;
    c.constraint_instance_id = instance;
    c.r_body_loc = M3D::Identity();
    c.t_body_loc = yaw(fr.true_yaw).transpose() * (LOC_WORLD - fr.true_pos);
    c.covariance = make_cov(0.0025, 0.0025);
    c.ts = ts;
    return c;
}

// --- Test 1: two sightings of one location close a drifted loop, pulling the
// (badly drifted) end back near the true end (the start). Distinct instance ids
// so neither sighting revises away the other.
static void test_location_closure() {
    Config cfg;
    cfg.use_location_constraints = true;
    cfg.key_pose_delta_trans = 0.5;
    SimplePGO pgo(cfg);

    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);
    for (int i = 0; i <= K; i++) {
        step(pgo, i * 0.1, f[i].odom_pos, yaw(f[i].odom_yaw));
        if (i == 0 || i == K)
            sight(pgo, f[i].odom_pos, yaw(f[i].odom_yaw), i * 0.1,
                  constraint_at("C", "C" + std::to_string(i), f[i], i * 0.1));
    }
    const auto& kps = pgo.keyPoses();
    double fed_err = (f[K].odom_pos - f[K].true_pos).norm();
    double cor_err = (kps.back().t_global - f[K].true_pos).norm();
    fprintf(stderr,
            "   true end=(%.1f,%.1f) fed end=(%.1f,%.1f) err %.1f m -> "
            "corrected=(%.1f,%.1f) err %.1f m\n",
            f[K].true_pos.x(), f[K].true_pos.y(), f[K].odom_pos.x(), f[K].odom_pos.y(),
            fed_err, kps.back().t_global.x(), kps.back().t_global.y(), cor_err);
    CHECK(fed_err > 20.0, "odometry drift opens the loop by >20 m");
    CHECK(cor_err < fed_err * 0.5,
          "location closure pulls the drifted end back to the true end by >50%");
}

// --- Test 2: the gravity anchor preserves the initial roll/pitch. Start with a
// 30 deg pitch; after constraint-driven optimization, keyframe 0's pitch is kept.
static double pitch_deg(const M3D& r) { return std::asin(-r(2, 0)) * 180.0 / M_PI; }

static void test_gravity_anchor() {
    Config cfg;
    cfg.use_location_constraints = true;
    cfg.key_pose_delta_trans = 0.5;
    SimplePGO pgo(cfg);

    const double pitch0 = 30.0 * M_PI / 180.0;
    M3D r0(Eigen::AngleAxisd(pitch0, V3D::UnitY()));  // 30 deg pitch about y

    // A drifted loop (so location closures actively pull on the graph) whose
    // first keyframe is pitched 30 deg. The anchor must keep that pitch.
    LoopFrame fr0{V3D(0, 0, 0), 0.0, V3D(0, 0, 0), 0.0};
    LoopFrame fr4{V3D(5, 5, 0), 0.0, V3D(5, 5, 0), 0.0};
    step(pgo, 0.0, V3D(0, 0, 0), r0);
    sight(pgo, V3D(0, 0, 0), r0, 0.0, constraint_at("G", "G0", fr0, 0.0));
    step(pgo, 1.0, V3D(10, 0, 0));
    step(pgo, 2.0, V3D(10, 12, 0));
    step(pgo, 3.0, V3D(0, 8, 0));
    step(pgo, 4.0, V3D(5, 5, 0), r0);
    sight(pgo, V3D(5, 5, 0), r0, 4.0, constraint_at("G", "G4", fr4, 4.0));

    double p_in = 30.0;
    double p_out = pitch_deg(pgo.keyPoses().front().r_global);
    fprintf(stderr, "   kf0 pitch in=%.4f out=%.4f deg\n", p_in, p_out);
    CHECK(std::abs(p_out - p_in) < 0.1,
          "gravity anchor keeps kf0 pitch within 0.1 deg of input");
}

// --- Test 3: revision supersedes a stale (wrong) committed constraint factor.
// A badly-wrong first sighting commits under instance "R0"; a corrective sighting
// reusing "R0" removes it; the final closure then uses the correct geometry, so
// the end is pulled back near the true end (a wrong, un-revised factor would drag
// the solution off).
static void test_revision() {
    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);

    auto run = [&](bool corrective) {
        Config cfg;
        cfg.use_location_constraints = true;
        cfg.key_pose_delta_trans = 0.5;
        SimplePGO pgo(cfg);
        for (int i = 0; i <= K; i++) {
            step(pgo, i * 0.1, f[i].odom_pos, yaw(f[i].odom_yaw));
            if (i == 0) {
                // Wrong first estimate (off by 15 m in body-y), instance "R0".
                LocationConstraintObs bad = constraint_at("R", "R0", f[0], 0.0);
                bad.t_body_loc += V3D(0, 15, 0);
                sight(pgo, f[0].odom_pos, yaw(f[0].odom_yaw), 0.0, bad);
            } else if (i == 1 && corrective) {
                // Corrective sighting reuses instance "R0" -> removes the stale one.
                sight(pgo, f[1].odom_pos, yaw(f[1].odom_yaw), 0.1,
                      constraint_at("R", "R0", f[1], 0.1));
            } else if (i == K) {
                // Final sighting, distinct instance -> closes the loop.
                sight(pgo, f[K].odom_pos, yaw(f[K].odom_yaw), i * 0.1,
                      constraint_at("R", "RK", f[K], i * 0.1));
            }
        }
        return (pgo.keyPoses().back().t_global - f[K].true_pos).norm();
    };

    double poisoned = run(false);
    double revised = run(true);
    fprintf(stderr, "   end error vs true: no-revision=%.1f m, with-revision=%.1f m\n",
            poisoned, revised);
    CHECK(revised < poisoned - 5.0,
          "revision removes the stale wrong constraint -> cleaner closure than without");
}

// --- Test 4: the per-keyframe gravity anchor keeps EVERY keyframe level through a
// loop closure (not just kf0). The keyframes drive on flat ground (zero
// roll/pitch); after the location closure corrects the drift, no inner keyframe
// should have tilted — tilt is what converts horizontal travel into vertical and
// corrupts z. Compares anchor off vs on.
static void test_per_keyframe_gravity() {
    const int K = 240;
    auto f = make_drifted_loop(K, 30.0, 0.0040);
    auto run = [&](bool per_kf) {
        Config cfg;
        cfg.use_location_constraints = true;
        cfg.key_pose_delta_trans = 0.5;
        cfg.gravity_anchor_per_keyframe = per_kf;
        SimplePGO pgo(cfg);
        for (int i = 0; i <= K; i++) {
            step(pgo, i * 0.1, f[i].odom_pos, yaw(f[i].odom_yaw));
            if (i == 0 || i == K)
                sight(pgo, f[i].odom_pos, yaw(f[i].odom_yaw), i * 0.1,
                      constraint_at("P", "P" + std::to_string(i), f[i], i * 0.1));
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
    fprintf(stderr, "== test_location_closure ==\n");
    test_location_closure();
    fprintf(stderr, "== test_gravity_anchor ==\n");
    test_gravity_anchor();
    fprintf(stderr, "== test_per_keyframe_gravity ==\n");
    test_per_keyframe_gravity();
    fprintf(stderr, "== test_revision ==\n");
    test_revision();
    if (g_failures) {
        fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
        return 1;
    }
    fprintf(stderr, "\nall assertions passed\n");
    return 0;
}
