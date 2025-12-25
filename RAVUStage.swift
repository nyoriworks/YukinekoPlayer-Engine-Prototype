import Metal
import simd

struct RAVUParamsSwift {
    var inputSize: SIMD2<Float>
    var outputSize: SIMD2<Float>
}

final class RAVUStage {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    
    // ★ ティアリング対策： 3枚のお皿
    private var texturePool: [MTLTexture?] = [nil, nil, nil]
    private var poolIndex = 0

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let function = library.makeFunction(name: "ravuUpscale") else {
            throw NSError(domain: "RAVU", code: -1, userInfo: [NSLocalizedDescriptionKey: "kernel missing"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    // process: ターゲットサイズへ美しく拡大
    func process(commandBuffer: MTLCommandBuffer,
                 source: MTLTexture,
                 targetWidth: Int,
                 targetHeight: Int) -> MTLTexture {
        
        // お皿ローテーション
        poolIndex = (poolIndex + 1) % 3
        
        // お皿の準備
        if texturePool[poolIndex] == nil ||
           texturePool[poolIndex]!.width != targetWidth ||
           texturePool[poolIndex]!.height != targetHeight {
            
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: source.pixelFormat, // 16Float維持
                width: targetWidth,
                height: targetHeight,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            texturePool[poolIndex] = device.makeTexture(descriptor: desc)
        }
        
        guard let dst = texturePool[poolIndex],
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return source
        }
        
        encoder.label = "RAVU (Love)"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(dst, index: 1)
        
        var params = RAVUParamsSwift(
            inputSize: SIMD2(Float(source.width), Float(source.height)),
            outputSize: SIMD2(Float(targetWidth), Float(targetHeight))
        )
        encoder.setBytes(&params, length: MemoryLayout<RAVUParamsSwift>.stride, index: 0)
        
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threads = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: dst.width, height: dst.height, depth: 1)
        
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        
        return dst
    }
}//
//  RAVUStage.swift
//  ViewArkPlayer
//
//  Created by Nyori on 2025/11/24.
//

