#include <metal_stdlib>
using namespace metal;

// 活性化関数
inline half activate(half x) {
    return max(0.0h, x); // ReLU
}

// ==========================================
// Layer 1: 1ch (Input) -> 24ch (Inter1)
// Kernel: 5x5
// ==========================================
kernel void espcn_lite_layer1(
    texture2d<half, access::sample> inTexture [[texture(0)]],
    texture2d_array<half, access::write> outTexture [[texture(1)]], // 24枚のレイヤー
    constant half* weights [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    // Conv1の設定
    const int in_ch = 1;
    const int out_ch = 24;
    const int k = 5; // 5x5
    const int pad = 2;
    
    // 重みデータの開始位置 (Layer1は最初なので0)
    int w_offset = 0;
    // バイアスの開始位置 (Weightの後ろ)
    int b_offset = w_offset + (out_ch * in_ch * k * k);

    // 24チャンネル分計算
    for (int oc = 0; oc < out_ch; oc++) {
        half sum = 0.0h;
        
        // 5x5 畳み込み
        for (int ky = -pad; ky <= pad; ky++) {
            for (int kx = -pad; kx <= pad; kx++) {
                // 入力は1枚だけ (coord::pixelモードでは+0.5でピクセル中心を指す)
                half val = inTexture.sample(s, float2(float(gid.x + kx) + 0.5f, float(gid.y + ky) + 0.5f)).r;
                
                // 重みのインデックス計算
                // [out_ch][in_ch][k][k]
                int w_idx = oc * (k*k) + (ky + pad) * k + (kx + pad);
                sum += val * weights[w_offset + w_idx];
            }
        }
        // バイアス加算 + ReLU
        sum += weights[b_offset + oc];
        outTexture.write(activate(sum), gid, oc); // Arrayの 'oc' 番目に書き込む
    }
}

// ==========================================
// Layer 2: 24ch (Inter1) -> 12ch (Inter2)
// Kernel: 3x3
// ==========================================
kernel void espcn_lite_layer2(
    texture2d_array<half, access::sample> inTexture [[texture(0)]],
    texture2d_array<half, access::write> outTexture [[texture(1)]], // 12枚のレイヤー
    constant half* weights [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);

    // Conv2の設定
    const int in_ch = 24;
    const int out_ch = 12;
    const int k = 3; // 3x3
    const int pad = 1;

    // オフセット計算 (Layer1の分を飛ばす)
    // Layer1 Total = (24*1*5*5) + 24 = 600 + 24 = 624
    int w_offset = 624;
    int b_offset = w_offset + (out_ch * in_ch * k * k);

    for (int oc = 0; oc < out_ch; oc++) {
        half sum = 0.0h;
        
        // 3x3 畳み込み (入力24枚すべてを足し合わせる)
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    // TextureArrayの 'ic' 番目をサンプリング (coord::pixelモードでは+0.5)
                    half val = inTexture.sample(s, float2(float(gid.x + kx) + 0.5f, float(gid.y + ky) + 0.5f), ic).r;
                    
                    int w_idx = oc * (in_ch * k * k) + ic * (k * k) + (ky + pad) * k + (kx + pad);
                    sum += val * weights[w_offset + w_idx];
                }
            }
        }
        sum += weights[b_offset + oc];
        outTexture.write(activate(sum), gid, oc);
    }
}

// ==========================================
// Layer 3: 12ch (Inter2) -> Output (PixelShuffle)
// Kernel: 3x3
// ==========================================
kernel void espcn_lite_layer3(
    texture2d_array<half, access::sample> inTexture [[texture(0)]],
    texture2d<half, access::write> outTexture [[texture(1)]], // 最終出力
    constant half* weights [[buffer(0)]],
    constant int& scaleFactor [[buffer(1)]], // x2 or x3
    uint2 gid [[thread_position_in_grid]]
) {
    // gid は「出力解像度」で動く (PixelShuffle済み座標)
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);

    const int in_ch = 12;
    const int k = 3;
    const int pad = 1;
    
    // 出力座標から入力座標(LR)への逆算
    uint2 srcPos = gid / scaleFactor;
    
    // PixelShuffle: どのチャネル(Phase)を使うか計算
    // 例(x2): (0,0)->ch0, (1,0)->ch1, (0,1)->ch2, (1,1)->ch3
    uint2 phase = gid % scaleFactor;
    int target_ch = phase.y * scaleFactor + phase.x; // これが出力チャネル番号(0 ~ scale^2-1)

    // オフセット計算
    // Layer1(624) + Layer2(24*12*9 + 12 = 2592 + 12 = 2604) = 3228
    int w_offset = 3228;
    int b_offset = w_offset + (scaleFactor * scaleFactor * in_ch * k * k);

    // Conv3計算 (必要なターゲットチャネル 1個分だけ計算する！)
    half sum = 0.0h;
    
    for (int ic = 0; ic < in_ch; ic++) {
        for (int ky = -pad; ky <= pad; ky++) {
            for (int kx = -pad; kx <= pad; kx++) {
                // coord::pixelモードでは+0.5でピクセル中心を指す
                half val = inTexture.sample(s, float2(float(srcPos.x + kx) + 0.5f, float(srcPos.y + ky) + 0.5f), ic).r;
                
                // target_ch 番目のフィルタを使う
                int w_idx = target_ch * (in_ch * k * k) + ic * (k * k) + (ky + pad) * k + (kx + pad);
                sum += val * weights[w_offset + w_idx];
            }
        }
    }
    
    // Layer3にReLUはない！
    sum += weights[b_offset + target_ch];
    
    // 書き込み
    outTexture.write(sum, gid);
}
