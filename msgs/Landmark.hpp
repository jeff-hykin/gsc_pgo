// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode-only C++ helper mirroring `dimos.navigation.jnav.msgs.Landmark`.
// Keep decode() in sync with Landmark.py's lcm_encode/lcm_decode.
//
// Wire format (big-endian), matching Landmark.py:
//
//   double ts
//   for each of (id, map_id, frame_id, kind):
//     uint32 len
//     bytes  utf-8 (no terminator)
//   double confidence
//   7×double pose: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w
//   double replacement            // milliseconds; 0 = additive (legacy bytes omit it)
//   6×double per-axis confidence: x, y, z, roll, pitch, yaw  (legacy bytes omit;
//                                 then each defaults to the coarse `confidence`)
//
// This class exposes the minimal interface lcm-cpp's templated subscribe needs
// (an instance decode(const void*, int, int)); it never encodes or publishes.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace jnav {

namespace landmark_detail {

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

}  // namespace landmark_detail

class Landmark {
public:
    double ts = 0.0;
    std::string id;
    std::string map_id;
    std::string frame_id;
    std::string kind;
    double confidence = 0.0;
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    double replacement = 0.0;  // ms
    // Per-axis confidence in [0, 1]; default to the coarse `confidence` for
    // legacy bytes that omit them.
    double confidence_x = 0.0, confidence_y = 0.0, confidence_z = 0.0;
    double confidence_roll = 0.0, confidence_pitch = 0.0, confidence_yaw = 0.0;

    // lcm-cpp calls this on the raw payload. Returns bytes consumed, or -1.
    int decode(const void* buf, int start, int maxlen) {
        const uint8_t* data = static_cast<const uint8_t*>(buf);
        int off = start;
        const int end = start + maxlen;
        auto need = [&](int n) { return off + n <= end; };

        if (!need(8)) return -1;
        ts = landmark_detail::read_double_be(data + off);
        off += 8;

        std::string* fields[4] = {&id, &map_id, &frame_id, &kind};
        for (auto* field : fields) {
            if (!need(4)) return -1;
            uint32_t len = landmark_detail::read_u32_be(data + off);
            off += 4;
            if (!need(static_cast<int>(len))) return -1;
            field->assign(reinterpret_cast<const char*>(data + off), len);
            off += static_cast<int>(len);
        }

        if (!need(8)) return -1;
        confidence = landmark_detail::read_double_be(data + off);
        off += 8;

        if (!need(56)) return -1;
        pos_x = landmark_detail::read_double_be(data + off + 0);
        pos_y = landmark_detail::read_double_be(data + off + 8);
        pos_z = landmark_detail::read_double_be(data + off + 16);
        quat_x = landmark_detail::read_double_be(data + off + 24);
        quat_y = landmark_detail::read_double_be(data + off + 32);
        quat_z = landmark_detail::read_double_be(data + off + 40);
        quat_w = landmark_detail::read_double_be(data + off + 48);
        off += 56;

        // Backward-compatible: legacy encodings omit the trailing replacement.
        if (need(8)) {
            replacement = landmark_detail::read_double_be(data + off);
            off += 8;
        } else {
            replacement = 0.0;
        }
        // Per-axis confidence (legacy bytes omit them -> default to confidence).
        if (need(48)) {
            confidence_x = landmark_detail::read_double_be(data + off + 0);
            confidence_y = landmark_detail::read_double_be(data + off + 8);
            confidence_z = landmark_detail::read_double_be(data + off + 16);
            confidence_roll = landmark_detail::read_double_be(data + off + 24);
            confidence_pitch = landmark_detail::read_double_be(data + off + 32);
            confidence_yaw = landmark_detail::read_double_be(data + off + 40);
            off += 48;
        } else {
            confidence_x = confidence_y = confidence_z = confidence;
            confidence_roll = confidence_pitch = confidence_yaw = confidence;
        }
        return off - start;
    }

    // Minimal stubs so the class satisfies lcm-cpp's message concept. The PGO
    // never publishes Landmarks, so these are unused.
    static const char* getTypeName() { return "jnav.Landmark"; }
    int getEncodedSize() const { return 0; }
};

}  // namespace jnav
