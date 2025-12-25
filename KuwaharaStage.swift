import Metal
import simd

struct KuwaharaParamsSwift {
    var resolution: SIMD2<Float>
    var radius: Int32
}

final class KuwaharaStage {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    
    // ティアリング対策：3枚のお皿
    private var texturePool: [MTLTexture?] = [nil, nil, nil]
    private var poolIndex = 0

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let function = library.makeFunction(name: "kuwahara") else {
            throw NSError(domain: "KuwaharaStage", code: -1, userInfo: [NSLocalizedDescriptionKey: "kernel missing"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func apply(commandBuffer: MTLCommandBuffer, source: MTLTexture, radius: Int = 1) -> MTLTexture {
        
        // 半径0なら何もしない
        if radius <= 0 { return source }
        
        poolIndex = (poolIndex + 1) % 3
        
        if texturePool[poolIndex] == nil ||
           texturePool[poolIndex]!.width != source.width ||
           texturePool[poolIndex]!.height != source.height {
            
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: source.pixelFormat,
                width: source.width,
                height: source.height,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            texturePool[poolIndex] = device.makeTexture(descriptor: desc)
        }
        
        guard let destination = texturePool[poolIndex],
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return source
        }
        
        encoder.label = "Kuwahara"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)
        
        // radius: 2〜4 くらいで調整してね！大きいほど「塗り」が強くなる。
        var params = KuwaharaParamsSwift(
            resolution: SIMD2(Float(source.width), Float(source.height)),
            radius: Int32(radius)
        )
        encoder.setBytes(&params, length: MemoryLayout<KuwaharaParamsSwift>.stride, index: 0)
        
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threads = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: source.width, height: source.height, depth: 1)
        
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        
        return destination
    }
}
