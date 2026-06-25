// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode-only C++ helper mirroring `dimos.navigation.jnav.msgs.LocationConstraint`.
// Keep decode() in sync with LocationConstraint.py's lcm_encode/lcm_decode.
//
// A LocationConstraint is the ready-made args for a GTSAM BetweenFactor: a
// relative-pose measurement from `frame_id` (the "from", enforced == body frame
// for now) to the location variable `to_id` (the "to"), plus its 6x6 covariance.
//
// Wire format (big-endian), matching LocationConstraint.py:
//
//   double ts
//   for each of (to_id, frame_id, constraint_instance_id):
//     uint32 len
//     bytes  utf-8 (no terminator)
//   7×double pose: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w  (from->to)
//   36×double covariance: row-major 6x6, GTSAM Pose3 tangent order [rot(3), trans(3)]
//
// This class exposes the minimal interface lcm-cpp's templated subscribe needs
// (an instance decode(const void*, int, int)); it never encodes or publishes.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace jnav {

namespace location_constraint_detail {

inline uint32_t read_u32_be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

inline double read_double_be(const uint8_t* p) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; i++) {
        bits = (bits << 8) | static_cast<uint64_t>(p[i]);
    }
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace location_constraint_detail

class LocationConstraint {
public:
    double ts = 0.0;
    std::string to_id;                  // the BetweenFactor "to" (location variable id)
    std::string frame_id;               // the BetweenFactor "from" frame (== body frame)
    std::string constraint_instance_id; // external instance id, for revision/removal
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    double covariance[36] = {0.0};      // row-major 6x6, tangent order [rot, trans]

    // lcm-cpp calls this on the raw payload. Returns bytes consumed, or -1.
    int decode(const void* buf, int start, int maxlen) {
        const uint8_t* data = static_cast<const uint8_t*>(buf);
        int off = start;
        const int end = start + maxlen;
        auto need = [&](int n) { return off + n <= end; };

        if (!need(8)) return -1;
        ts = location_constraint_detail::read_double_be(data + off);
        off += 8;

        std::string* fields[3] = {&to_id, &frame_id, &constraint_instance_id};
        for (auto* field : fields) {
            if (!need(4)) return -1;
            uint32_t len = location_constraint_detail::read_u32_be(data + off);
            off += 4;
            if (!need(static_cast<int>(len))) return -1;
            field->assign(reinterpret_cast<const char*>(data + off), len);
            off += static_cast<int>(len);
        }

        if (!need(56)) return -1;
        pos_x = location_constraint_detail::read_double_be(data + off + 0);
        pos_y = location_constraint_detail::read_double_be(data + off + 8);
        pos_z = location_constraint_detail::read_double_be(data + off + 16);
        quat_x = location_constraint_detail::read_double_be(data + off + 24);
        quat_y = location_constraint_detail::read_double_be(data + off + 32);
        quat_z = location_constraint_detail::read_double_be(data + off + 40);
        quat_w = location_constraint_detail::read_double_be(data + off + 48);
        off += 56;

        if (!need(36 * 8)) return -1;
        for (int i = 0; i < 36; i++) {
            covariance[i] = location_constraint_detail::read_double_be(data + off + i * 8);
        }
        off += 36 * 8;

        return off - start;
    }

    // Minimal stubs so the class satisfies lcm-cpp's message concept. The PGO
    // never publishes LocationConstraints, so these are unused.
    static const char* getTypeName() { return "jnav.LocationConstraint"; }
    int getEncodedSize() const { return 0; }
};

}  // namespace jnav
