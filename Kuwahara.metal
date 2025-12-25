#include <metal_stdlib>
using namespace metal;

struct KuwaharaParams {
    float2 resolution;
    int radius; // 半径 (2〜4くらいがおすすめ)
};

kernel void kuwahara(texture2d<float, access::sample> inTex [[texture(0)]],
                     texture2d<float, access::write> outTex [[texture(1)]],
                     constant KuwaharaParams &params [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
    float2 uv = (float2(gid) + 0.5) / params.resolution;
    float2 px = 1.0 / params.resolution;
    
    int radius = params.radius;
    float n = float((radius + 1) * (radius + 1));

    // 4つの領域の 平均(m) と 分散(s) を計算するための変数
    // [0]: 左上, [1]: 右上, [2]: 左下, [3]: 右下
    float3 m[4] = {0.0, 0.0, 0.0, 0.0};
    float3 s_val[4] = {0.0, 0.0, 0.0, 0.0};

    // --- 4つの領域をループして統計を取る ---
    // (ここが少し重いけど、魔法のため！)
    
    // Region 0 (左上)
    for (int j = -radius; j <= 0; ++j)  {
        for (int i = -radius; i <= 0; ++i)  {
            float3 c = inTex.sample(s, uv + float2(i, j) * px).rgb;
            m[0] += c;
            s_val[0] += c * c;
        }
    }

    // Region 1 (右上)
    for (int j = -radius; j <= 0; ++j)  {
        for (int i = 0; i <= radius; ++i)  {
            float3 c = inTex.sample(s, uv + float2(i, j) * px).rgb;
            m[1] += c;
            s_val[1] += c * c;
        }
    }

    // Region 2 (左下)
    for (int j = 0; j <= radius; ++j)  {
        for (int i = -radius; i <= 0; ++i)  {
            float3 c = inTex.sample(s, uv + float2(i, j) * px).rgb;
            m[2] += c;
            s_val[2] += c * c;
        }
    }

    // Region 3 (右下)
    for (int j = 0; j <= radius; ++j)  {
        for (int i = 0; i <= radius; ++i)  {
            float3 c = inTex.sample(s, uv + float2(i, j) * px).rgb;
            m[3] += c;
            s_val[3] += c * c;
        }
    }

    // --- 最小分散（一番色が平坦な領域）を選ぶ ---
    float minSigma = 1e+2;
    float3 finalColor = float3(0.0);

    for (int k = 0; k < 4; ++k) {
        m[k] /= n;
        s_val[k] = abs(s_val[k] / n - m[k] * m[k]);
        float sigma2 = s_val[k].r + s_val[k].g + s_val[k].b;
        
        if (sigma2 < minSigma) {
            minSigma = sigma2;
            finalColor = m[k];
        }
    }

    outTex.write(float4(finalColor, 1.0), gid);
}
