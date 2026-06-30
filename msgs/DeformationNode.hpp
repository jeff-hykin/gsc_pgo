// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Typed C++ helper mirroring the Python `dimos.msgs.nav_msgs.DeformationNode`.
// Keep encode() in sync with DeformationNode.py.lcm_decode.
//
// One pose-graph keyframe, published individually (not batched). The producer
// emits one on keyframe creation and again whenever the optimizer moves it
// (same `id`). A recording captures the stream so a transform lookup can later
// correct tf for loop closure.
//
// Wire format (big-endian):
//
//   uint64 id          // stable random id for this keyframe
//   uint64 tf_id       // fnv1a_64(frame_from + "|" + frame_to)
//   double pose_ts     // seconds since epoch
//   uint32 frame_id_len
//   bytes  frame_id    // utf-8, no terminator
//   7×double pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <lcm/lcm-cpp.hpp>

namespace dimos {

namespace deformation_node_detail {

inline void write_u32_be(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >>  8) & 0xFF));
    out.push_back(static_cast<uint8_t>( v        & 0xFF));
}

inline void write_u64_be(std::vector<uint8_t>& out, uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((v >> shift) & 0xFF));
    }
}

inline void write_double_be(std::vector<uint8_t>& out, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    write_u64_be(out, bits);
}

}  // namespace deformation_node_detail

// 64-bit FNV-1a hash. Must match DeformationNode.py's fnv1a_64 so tf_id
// filtering agrees across the wire.
inline uint64_t fnv1a_64(const std::string& text) {
    constexpr uint64_t kOffsetBasis = 0xCBF29CE484222325ULL;
    constexpr uint64_t kPrime = 0x100000001B3ULL;
    uint64_t digest = kOffsetBasis;
    for (unsigned char byte : text) {
        digest = (digest ^ byte) * kPrime;
    }
    return digest;
}

// tf_id for a transform edge: fnv1a_64(frame_from + "|" + frame_to).
inline uint64_t tf_id_for(const std::string& frame_from, const std::string& frame_to) {
    return fnv1a_64(frame_from + "|" + frame_to);
}

class DeformationNode {
public:
    DeformationNode(uint64_t id, uint64_t tf_id, double pose_ts, std::string frame_id,
                    double pos_x, double pos_y, double pos_z,
                    double quat_x, double quat_y, double quat_z, double quat_w)
        : id_(id), tf_id_(tf_id), pose_ts_(pose_ts), frame_id_(std::move(frame_id)),
          pos_x_(pos_x), pos_y_(pos_y), pos_z_(pos_z),
          quat_x_(quat_x), quat_y_(quat_y), quat_z_(quat_z), quat_w_(quat_w) {}

    std::vector<uint8_t> encode() const {
        using namespace deformation_node_detail;
        std::vector<uint8_t> out;
        out.reserve(20 + frame_id_.size() + 56);
        write_u64_be(out, id_);
        write_u64_be(out, tf_id_);
        write_double_be(out, pose_ts_);
        write_u32_be(out, static_cast<uint32_t>(frame_id_.size()));
        out.insert(out.end(), frame_id_.begin(), frame_id_.end());
        write_double_be(out, pos_x_);
        write_double_be(out, pos_y_);
        write_double_be(out, pos_z_);
        write_double_be(out, quat_x_);
        write_double_be(out, quat_y_);
        write_double_be(out, quat_z_);
        write_double_be(out, quat_w_);
        return out;
    }

    int publish(lcm::LCM& lcm, const std::string& channel) const {
        std::vector<uint8_t> bytes = encode();
        return lcm.publish(channel, bytes.data(), static_cast<int>(bytes.size()));
    }

private:
    uint64_t id_;
    uint64_t tf_id_;
    double pose_ts_;
    std::string frame_id_;
    double pos_x_, pos_y_, pos_z_;
    double quat_x_, quat_y_, quat_z_, quat_w_;
};

}  // namespace dimos
