#include "cumsum.cuh"

#define CUDA_CUMSUM_BLOCK_SIZE 256
#define CUDA_CUMSUM_UNROLL 4

// Optimized cumsum kernel using register blocking (unroll=4).
// Each thread processes 4 elements sequentially in registers before
// participating in the shared-memory scan, reducing __syncthreads()
// overhead by ~4x and improving instruction-level parallelism.
// Based on upstream llama.cpp PR #18343.
static __global__ void cumsum_f32_kernel(
        const float * src, float * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s00,  const int64_t s01,  const int64_t s02,  const int64_t s03,
        const int64_t d0,   const int64_t d1,   const int64_t d2,   const int64_t d3) {
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const float * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    float * dst_row = dst + i1 * d1 + i2 * d2 + i3 * d3;

    extern __shared__ float s_scan[];

    float carry = 0.0f;
    // Each iteration processes blockDim.x * CUDA_CUMSUM_UNROLL elements
    for (int64_t start = 0; start < ne00; start += blockDim.x * CUDA_CUMSUM_UNROLL) {
        // Phase 1: Thread-local sequential prefix sum over CUDA_CUMSUM_UNROLL elements
        const int64_t base_idx = start + threadIdx.x * CUDA_CUMSUM_UNROLL;
        float local[CUDA_CUMSUM_UNROLL];

        local[0] = (base_idx < ne00) ? src_row[base_idx * s00] : 0.0f;
#pragma unroll
        for (int j = 1; j < CUDA_CUMSUM_UNROLL; j++) {
            float v = (base_idx + j < ne00) ? src_row[(base_idx + j) * s00] : 0.0f;
            local[j] = local[j - 1] + v;
        }

        // Phase 2: Inter-thread inclusive scan on per-thread totals (last element = thread sum)
        float thread_sum = local[CUDA_CUMSUM_UNROLL - 1];
        s_scan[threadIdx.x] = thread_sum;
        __syncthreads();

        for (int offset = 1; offset < (int)blockDim.x; offset <<= 1) {
            float add = 0.0f;
            if (threadIdx.x >= offset) {
                add = s_scan[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x >= offset) {
                s_scan[threadIdx.x] += add;
            }
            __syncthreads();
        }

        // Phase 3: Write back with carry + exclusive prefix from other threads
        // s_scan[threadIdx.x] is inclusive scan of thread sums
        // We need the exclusive prefix: sum of all threads before this one
        float thread_offset = (threadIdx.x > 0 ? s_scan[threadIdx.x - 1] : 0.0f) + carry;

#pragma unroll
        for (int j = 0; j < CUDA_CUMSUM_UNROLL; j++) {
            if (base_idx + j < ne00) {
                dst_row[(base_idx + j) * d0] = local[j] + thread_offset;
            }
        }

        // Broadcast carry to all threads via shared memory
        __syncthreads();
        if (threadIdx.x == 0) {
            s_scan[0] = carry + s_scan[blockDim.x - 1];
        }
        __syncthreads();
        carry = s_scan[0];
    }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Each thread processes CUDA_CUMSUM_UNROLL elements, so we need
    // ceil(ne00 / CUDA_CUMSUM_UNROLL) threads, rounded up to a power of 2.
    const int64_t threads_needed = (src0->ne[0] + CUDA_CUMSUM_UNROLL - 1) / CUDA_CUMSUM_UNROLL;
    int block_size = WARP_SIZE;
    while (block_size < threads_needed && block_size < CUDA_CUMSUM_BLOCK_SIZE) {
        block_size <<= 1;
    }

    dim3 grid_dims(src0->ne[1], src0->ne[2], src0->ne[3]);
    cumsum_f32_kernel<<<grid_dims, block_size, block_size * sizeof(float), ctx.stream()>>>(
        (const float *) src0->data,
        (float *) dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0] / sizeof(float), src0->nb[1] / sizeof(float), src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
        dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), dst->nb[3] / sizeof(float));
}
