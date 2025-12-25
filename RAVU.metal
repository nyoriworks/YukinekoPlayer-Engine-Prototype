#include <metal_stdlib>
using namespace metal;

struct RAVUParams {
    float2 inputSize;
    float2 outputSize;
};

// 輝度計算 (BT.709係数: 人間の目の感度に合わせる愛の重み)
inline float getLuma(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

// ★ 秘伝のタレ 1： 「S字カーブ」関数
// ただの線形補間ではなく、このカーブを通すことで
// ボケ味を排除し、エッジを「キリッ」と立たせる！
inline float s_curve(float x) {
    return x * x * (3.0 - 2.0 * x);
}

kernel void ravuUpscale(texture2d<float, access::sample> inTex [[texture(0)]],
                        texture2d<float, access::write> outTex [[texture(1)]],
                        constant RAVUParams &params [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5) / params.outputSize;
    float2 px = 1.0 / params.inputSize;

    // 座標計算: 中心を合わせる
    float2 pos = uv * params.inputSize - 0.5;
    float2 f = fract(pos); // 小数部分 (0.0 〜 1.0)
    
    // --- 6点サンプリング (RAVUの基本陣形) ---
    //  a b
    //  c d
    //  e f
    
    // 基準点 (c) のテクスチャ座標
    float2 tc_c = floor(pos) * px + px * 0.5;
    
    // 周辺ピクセルを取得
    // (Linearサンプラーを使うことで、GPUのハードウェア補間を少し借りて高速化！)
    float3 c = inTex.sample(s, tc_c).rgb;
    float3 d = inTex.sample(s, tc_c + float2(px.x, 0)).rgb;
    float3 a = inTex.sample(s, tc_c + float2(0, -px.y)).rgb;
    float3 b = inTex.sample(s, tc_c + float2(px.x, -px.y)).rgb;
    float3 e = inTex.sample(s, tc_c + float2(0, px.y)).rgb;
    float3 f_pix = inTex.sample(s, tc_c + float2(px.x, px.y)).rgb;

    // --- 重み計算 (ここが魔法！) ---
    
    // 輝度を抽出
    float lc = getLuma(c); float ld = getLuma(d);
    float la = getLuma(a); float lb = getLuma(b);
    float le = getLuma(e); float lf = getLuma(f_pix);

    // 縦方向の勾配 (Gradient) を解析
    // 「色がどれくらい激しく変化しているか」を見る
    float grad_y_c = abs(lc - la) + abs(lc - le);
    float grad_y_d = abs(ld - lb) + abs(ld - lf);
    
    // 重み係数 (変化が激しいところは混ぜない！)
    float w_y_c = 1.0 - saturate(grad_y_c * 2.5); // 2.5 はキレ味調整用のタレ
    float w_y_d = 1.0 - saturate(grad_y_d * 2.5);

    // 縦方向の補間 (S字カーブ適用)
    float sy = s_curve(f.y);
    
    // 各列での予測色
    // 単純なmixではなく、重み(w_y)を使って「エッジを避ける」ように混ぜる
    float3 col_c_col = mix(c, mix(a, e, 0.5 * f.y), sy * (1.0 - w_y_c));
    float3 col_d_col = mix(d, mix(b, f_pix, 0.5 * f.y), sy * (1.0 - w_y_d));

    // 横方向の補間
    float sx = s_curve(f.x);
    float3 finalColor = mix(col_c_col, col_d_col, sx);

    // --- ★ 秘伝のタレ 2： アンチリンギング (Clamping) ---
    // どんなに計算しても、元の4点(a,b,c,d)の最大・最小を超えてはいけない。
    // これが「絶対に白いフチが出ない」理由！
    float3 minColor = min(min(c, d), min(a, e)); // 周辺の最小値
    float3 maxColor = max(max(c, d), max(a, e)); // 周辺の最大値
    
    // 少しだけ範囲を広げる(Relax)ことで、自然な階調を残す
    minColor -= 0.002;
    maxColor += 0.002;

    finalColor = clamp(finalColor, minColor, maxColor);

    outTex.write(float4(finalColor, 1.0), gid);
}//
//  RAVU.metal
//  ViewArkPlayer
//
//  Created by Nyori on 2025/11/24.
//

