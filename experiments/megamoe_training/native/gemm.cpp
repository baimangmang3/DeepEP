// BF16 training GEMMs for Ascend 950, using the pinned CATLASS implementation.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include <cstdint>
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

namespace {
template<class LA, class LB, class C>
int Run(void* a, void* b, void* c, uint32_t m, uint32_t n, uint32_t k,
        void* stream, uint32_t cores) {
    using namespace Catlass;
    using A = bfloat16_t;
    using B = bfloat16_t;
    using LC = layout::RowMajor;
    using Arch = Arch::Ascend950;
    using Policy = Gemm::MmadPingpong<Arch, true, false>;
    using L1 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
    using L0 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
    using Copy = Gemm::Tile::PackedTileCopyTla<Arch, A, LA, B, LB, C, LC>;
    using Block = Gemm::Block::BlockMmadTla<Policy, L1, L0, A, B, C, void, Copy>;
    using Schedule = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using Kernel = Gemm::Kernel::BasicMatmulTla<Block, void, Schedule>;
    using Device = Gemm::Device::DeviceGemm<Kernel>;
    typename Kernel::Arguments args{
        GemmCoord{m, n, k}, static_cast<uint8_t*>(a), tla::MakeLayout<A, LA>(m, k),
        static_cast<uint8_t*>(b), tla::MakeLayout<B, LB>(k, n),
        static_cast<uint8_t*>(c), tla::MakeLayout<C, LC>(m, n), nullptr};
    Device op;
    if (op.CanImplement(args) != Status::kSuccess || op.GetWorkspaceSize(args) != 0) return 1;
    op.Initialize(args);
    uint32_t tiles = ((m + 127) / 128) * ((n + 255) / 256);
    return static_cast<int>(op(static_cast<aclrtStream>(stream), tiles < cores ? tiles : cores));
}

template<class C>
int Select(void* a, void* b, void* c, uint32_t m, uint32_t n, uint32_t k,
           int ta, int tb, void* stream, uint32_t cores) {
    using R = Catlass::layout::RowMajor;
    using T = Catlass::layout::ColumnMajor;
    if (ta && tb) return Run<T, T, C>(a,b,c,m,n,k,stream,cores);
    if (ta) return Run<T, R, C>(a,b,c,m,n,k,stream,cores);
    if (tb) return Run<R, T, C>(a,b,c,m,n,k,stream,cores);
    return Run<R, R, C>(a,b,c,m,n,k,stream,cores);
}
}

extern "C" int megamoe_train_gemm(void* a, void* b, void* c,
    uint32_t m, uint32_t n, uint32_t k, int trans_a, int trans_b,
    int output_fp32, void* stream, uint32_t cores) {
    if (!a || !b || !c || !m || !n || !k || !cores) return 2;
    if (output_fp32) return Select<float>(a,b,c,m,n,k,trans_a,trans_b,stream,cores);
    return Select<bfloat16_t>(a,b,c,m,n,k,trans_a,trans_b,stream,cores);
}

extern "C" const char* megamoe_train_backend() {
    return "CATLASS-Ascend950-BF16-FP32acc-basic-v1";
}
