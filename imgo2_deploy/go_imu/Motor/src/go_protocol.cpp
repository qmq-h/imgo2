#include "go_protocol.h"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace go
{
    namespace
    {
        constexpr double r = static_cast<double>(kGearRatio), twoPi = 6.2831853071795864769;
        double clip(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }
        void put16(uint8_t *p, uint16_t v)
        {
            p[0] = uint8_t(v);
            p[1] = uint8_t(v >> 8);
        }
        void put32(uint8_t *p, uint32_t v)
        {
            for (int i = 0; i < 4; ++i)
                p[i] = uint8_t(v >> (8 * i));
        }
        uint16_t get16(const uint8_t *p) { return uint16_t(uint16_t(p[0]) | uint16_t(p[1]) << 8); }
        uint32_t get32(const uint8_t *p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
        // Explicit two's-complement decoding, independent of unsigned-to-signed casts.
        double signed16(const uint8_t *p)
        {
            auto v = get16(p);
            return v < 32768 ? v : double(v) - 65536;
        }
        double signed32(const uint8_t *p)
        {
            auto v = get32(p);
            return v < 0x80000000u ? v : double(v) - 4294967296.0;
        }
    }
    uint16_t crc16(const uint8_t *data, size_t len, uint16_t crc)
    {
        while (len--)
        {
            crc = uint16_t(crc ^ *data++);
            for (int i = 0; i < 8; ++i)
                crc = uint16_t((crc >> 1) ^ ((crc & 1) ? 0x8408 : 0));
        }
        return crc;
    }
    const char *motorErrorText(uint8_t e)
    {
        const char *names[] = {"normal", "overheat", "overcurrent", "overvoltage", "encoder fault", "bus undervoltage", "winding overheat"};
        return e < 7 ? names[e] : "unknown fault";
    }
    bool packCommand(const Command &c, uint8_t *out)
    {
        if (!out)
            return false;
        std::memset(out, 0, kCmdLen);
        if (c.id > 15 || uint8_t(c.mode) > 2 || !std::isfinite(c.q) || !std::isfinite(c.dq) ||
            !std::isfinite(c.tau) || !std::isfinite(c.kp) || !std::isfinite(c.kd))
            return false;
        out[0] = kCmdHead0;
        out[1] = kHead1;
        out[2] = uint8_t(c.id | (uint8_t(c.mode) << 4));
        // 对外统一输出轴语义：输出端力矩 = 转子力矩 × 6.33（理想减速器）。
        // 所以前馈力矩发送时除以 6.33；输出端角速度发送时乘以 6.33。
        auto tau = int16_t(clip(double(c.tau) / r, -127.99, 127.99) * 256);
        auto vel = int16_t(clip(clip(double(c.dq) * r, -804, 804) / twoPi * 256, -32768, 32767));
        auto pos = int32_t(clip(clip(double(c.q) * r, -411774, 411774) / twoPi * 32768, -2147483648.0, 2147483647.0));
        put16(out + 3, uint16_t(tau));
        put16(out + 5, uint16_t(vel));
        put32(out + 7, uint32_t(pos));
        // 位置/速度误差在转子侧扩大 6.33 倍，力矩换回转子侧缩小 6.33 倍，
        // 因此输出端 PD 增益转换为转子增益时都要除以 6.33²。
        put16(out + 11, uint16_t(clip(double(c.kp) / (r * r), 0, 25.599) * 1280));
        put16(out + 13, uint16_t(clip(double(c.kd) / (r * r), 0, 25.599) * 1280));
        put16(out + 15, crc16(out, 15));
        return true;
    }
    bool unpackFeedback(const uint8_t *in, size_t len, Feedback &out)
    {
        out.valid = false;
        if (!in || len != kReplyLen || in[0] != kReplyHead0 || in[1] != kHead1 ||
            crc16(in, 14) != get16(in + 14) || (in[2] & 15) > 14 || (in[2] >> 4) > 2)
            return false;
        out.id = in[2] & 15;
        out.mode = MotorMode(in[2] >> 4);
        // 反馈执行反方向换算；tau 是理想减速比下的输出轴力矩估计值。
        out.tau = float(signed16(in + 3) / 256 * r);
        out.dq = float(signed16(in + 5) / 256 * twoPi / r);
        out.q = float(signed32(in + 7) / 32768 * twoPi / r);
        out.temp = int8_t(in[11] < 128 ? int(in[11]) : int(in[11]) - 256);
        auto flags = get16(in + 12);
        out.error = flags & 7;
        out.force = (flags >> 3) & 4095;
        out.valid = true;
        return true;
    }
    bool findFrame(uint8_t *buf, size_t &len, size_t frameLen)
    {
        if (!buf)
        {
            len = 0;
            return false;
        }
        for (size_t i = 0; i + 1 < len; ++i)
            if (buf[i] == kReplyHead0 && buf[i + 1] == kHead1)
            {
                std::memmove(buf, buf + i, len - i);
                len -= i;
                return len >= frameLen;
            }
        bool tail = len && buf[len - 1] == kReplyHead0;
        len = tail ? 1 : 0;
        if (tail)
            buf[0] = kReplyHead0;
        return false;
    }
}
