/**
 * VP9 Loop Filter — Vulkan compute shader
 *
 * Applies VP9's adaptive deblocking filter along horizontal
 * and vertical block boundaries.
 *
 * 2D dispatch: X covers the pixel position along the boundary, Y selects
 * which 8-pixel boundary is filtered — one thread per (pixel, boundary)
 * pair, so the whole frame is filtered in a single well-occupied dispatch.
 */
#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Frame {
    uint8_t pixels[];
};

layout(push_constant) uniform PushConst {
    uint  frame_width;
    uint  frame_height;
    uint  stride;
    uint  filter_level;   /* 0–63 */
    uint  sharpness;      /* 0–7  */
    uint  pass;           /* 0=horizontal, 1=vertical */
} pc;

/* ── VP9 filter threshold derivation ───────────────────────────────────── */
uint limit_from_level(uint lvl, uint sharp)
{
    uint limit = lvl;
    if (sharp > 0u) {
        limit >>= (sharp > 4u) ? 2u : 1u;
        limit = min(limit, 9u - sharp);
    }
    return max(limit, 1u);
}

/* ── 4-tap simple filter ────────────────────────────────────────────────── */
void filter4(uint p1_off, uint p0_off, uint q0_off, uint q1_off, uint threshold)
{
    int p1 = int(pixels[p1_off]);
    int p0 = int(pixels[p0_off]);
    int q0 = int(pixels[q0_off]);
    int q1 = int(pixels[q1_off]);

    /* Flatness check */
    if (abs(p0 - q0) * 2 + abs(p1 - q1) / 2 > int(threshold)) return;

    int delta = clamp((3 * (q0 - p0) + 4) >> 3, -int(threshold), int(threshold));

    pixels[p0_off] = uint8_t(clamp(p0 + delta,     0, 255));
    pixels[q0_off] = uint8_t(clamp(q0 - delta,     0, 255));
    pixels[p1_off] = uint8_t(clamp(p1 + delta / 2, 0, 255));
    pixels[q1_off] = uint8_t(clamp(q1 - delta / 2, 0, 255));
}

void main()
{
    uint threshold = limit_from_level(pc.filter_level, pc.sharpness);

    if (pc.pass == 0u) {
        /* Horizontal pass — filter along horizontal boundaries */
        uint x = gl_GlobalInvocationID.x;
        uint y = (gl_GlobalInvocationID.y + 1u) * 8u;
        if (x >= pc.frame_width || y >= pc.frame_height) return;
        filter4(
            (y - 2u) * pc.stride + x,
            (y - 1u) * pc.stride + x,
            (y + 0u) * pc.stride + x,
            (y + 1u) * pc.stride + x,
            threshold
        );
    } else {
        /* Vertical pass — filter along vertical boundaries */
        uint y = gl_GlobalInvocationID.x;
        uint x = (gl_GlobalInvocationID.y + 1u) * 8u;
        if (y >= pc.frame_height || x >= pc.frame_width) return;
        filter4(
            y * pc.stride + x - 2u,
            y * pc.stride + x - 1u,
            y * pc.stride + x + 0u,
            y * pc.stride + x + 1u,
            threshold
        );
    }
}
