// Training-only BF16 pointwise candidates. No communication or GEMM is fused here.
// Saved GU and unweighted Z remain unchanged for backward, including zero gates.
#include "basic_api/kernel_basic_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#include <cstdint>

using namespace AscendC;

namespace {
constexpr uint32_t kRows = 8;
constexpr uint32_t kCols = 256;
constexpr uint32_t kElements = kRows * kCols;
constexpr Reg::CastTrait kToFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait kToBfloat = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// Op0: A=BF16(silu(FP32(G))*FP32(U)), input GU[rows,2*width].
// Op1: dGU from rounded BF16 dA and saved rounded BF16 GU, exact P0 operation order.
// Op2: output=BF16(FP32(input)*p), p is FP32[rows].
template<int Op>
__aicore__ inline void VectorTile(__ubuf__ bfloat16_t* g, __ubuf__ bfloat16_t* u,
    __ubuf__ bfloat16_t* da, __ubuf__ float* p, __ubuf__ bfloat16_t* out0,
    __ubuf__ bfloat16_t* out1, uint16_t rows, uint16_t cols) {
    constexpr uint16_t vectorElements = VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t vectors = cols / vectorElements; // host requires native 128 multiples
    __VEC_SCOPE__ {
        Reg::RegTensor<bfloat16_t> gb, ub, db, result;
        Reg::RegTensor<float> gf, uf, df, weight, neg, denominator, s, one, tmp0, tmp1, value;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate(one, 1.0F);
        for (uint16_t r = 0; r < rows; ++r) {
            if constexpr (Op == 2) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight, p+r);
            }
            for (uint16_t v = 0; v < vectors; ++v) {
                const uint32_t offset = r * cols + v * vectorElements;
                Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gb, g+offset);
                Reg::Cast<float, bfloat16_t, kToFloat>(gf, gb, mask);
                if constexpr (Op == 2) {
                    Reg::Mul(value, gf, weight, mask);
                } else {
                    Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(ub, u+offset);
                    Reg::Cast<float, bfloat16_t, kToFloat>(uf, ub, mask);
                    Reg::Muls(neg, gf, -1.0F, mask);
                    Reg::Exp(denominator, neg, mask);
                    Reg::Adds(denominator, denominator, 1.0F, mask);
                    if constexpr (Op == 0) {
                        // No clamp, no preweighting, no intermediate BF16 cast.
                        Reg::Div(tmp0, gf, denominator, mask);
                        Reg::Mul(value, tmp0, uf, mask);
                    } else {
                        Reg::Div(s, one, denominator, mask);
                        Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(db, da+offset);
                        Reg::Cast<float, bfloat16_t, kToFloat>(df, db, mask);
                        // dG = (((dA*U)*s)*(1 + G*(1-s))). Separate FP32 operations.
                        Reg::Sub(tmp0, one, s, mask);
                        Reg::Mul(tmp0, gf, tmp0, mask);
                        Reg::Adds(tmp0, tmp0, 1.0F, mask);
                        Reg::Mul(tmp1, df, uf, mask);
                        Reg::Mul(tmp1, tmp1, s, mask);
                        Reg::Mul(value, tmp1, tmp0, mask);
                    }
                }
                Reg::Cast<bfloat16_t, float, kToBfloat>(result, value, mask);
                Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(out0+offset, result, mask);
                if constexpr (Op == 1) {
                    // dU = (dA*G)*s, independently rounded to BF16.
                    Reg::Mul(tmp0, df, gf, mask);
                    Reg::Mul(value, tmp0, s, mask);
                    Reg::Cast<bfloat16_t, float, kToBfloat>(result, value, mask);
                    Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(out1+offset, result, mask);
                }
            }
        }
    }
}

template<int Op>
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void Pointwise(GM_ADDR input, GM_ADDR da, GM_ADDR p,
    GM_ADDR output, uint32_t rows, uint32_t width) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> gBuf, uBuf, dBuf, pBuf, out0Buf, out1Buf;
    pipe.InitBuffer(gBuf, kElements*sizeof(bfloat16_t));
    pipe.InitBuffer(out0Buf, kElements*sizeof(bfloat16_t));
    if constexpr (Op != 2) pipe.InitBuffer(uBuf, kElements*sizeof(bfloat16_t));
    if constexpr (Op == 1) {
        pipe.InitBuffer(dBuf, kElements*sizeof(bfloat16_t));
        pipe.InitBuffer(out1Buf, kElements*sizeof(bfloat16_t));
    }
    if constexpr (Op == 2) pipe.InitBuffer(pBuf, 32);
    auto g = gBuf.Get<bfloat16_t>();
    auto out0 = out0Buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> u, d, out1;
    LocalTensor<float> weights;
    if constexpr (Op != 2) u = uBuf.Get<bfloat16_t>();
    if constexpr (Op == 1) { d = dBuf.Get<bfloat16_t>(); out1 = out1Buf.Get<bfloat16_t>(); }
    if constexpr (Op == 2) weights = pBuf.Get<float>();
    GlobalTensor<bfloat16_t> inGm, dGm, outGm;
    GlobalTensor<float> pGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    if constexpr (Op == 1) dGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(da));
    if constexpr (Op == 2) pGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
    const uint32_t nTiles = (width+kCols-1)/kCols;
    const uint64_t tiles = ((static_cast<uint64_t>(rows)+kRows-1)/kRows)*nTiles;
    const uint32_t inStride = (Op == 2 ? width : 2*width);
    const uint32_t outStride = (Op == 1 ? 2*width : width);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (uint64_t tile=GetBlockIdx(); tile<tiles; tile+=GetBlockNum()) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        const uint32_t r0 = tile/nTiles*kRows, c0 = tile%nTiles*kCols;
        const uint16_t nr = rows-r0<kRows ? rows-r0 : kRows;
        const uint16_t nc = width-c0<kCols ? width-c0 : kCols;
        const uint64_t inputOffset = static_cast<uint64_t>(r0)*inStride+c0;
        DataCopyExtParams load{nr, static_cast<uint32_t>(nc*sizeof(bfloat16_t)),
            static_cast<uint32_t>((inStride-nc)*sizeof(bfloat16_t)), 0, 0};
        DataCopyPadExtParams<bfloat16_t> noPad{false,0,0,0};
        DataCopyPad(g, inGm[inputOffset], load, noPad);
        if constexpr (Op != 2) DataCopyPad(u, inGm[inputOffset+width], load, noPad);
        if constexpr (Op == 1) {
            DataCopyExtParams loadD{nr, static_cast<uint32_t>(nc*sizeof(bfloat16_t)),
                static_cast<uint32_t>((width-nc)*sizeof(bfloat16_t)), 0, 0};
            DataCopyPad(d, dGm[static_cast<uint64_t>(r0)*width+c0], loadD, noPad);
        }
        if constexpr (Op == 2) {
            DataCopyPad(weights, pGm[r0], {1,static_cast<uint32_t>(nr*sizeof(float)),0,0,0}, {false,0,0,0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        __ubuf__ bfloat16_t* up = nullptr;
        __ubuf__ bfloat16_t* dp = nullptr;
        __ubuf__ bfloat16_t* o1p = nullptr;
        __ubuf__ float* wp = nullptr;
        if constexpr (Op != 2) up = reinterpret_cast<__ubuf__ bfloat16_t*>(u.GetPhyAddr());
        if constexpr (Op == 1) {
            dp = reinterpret_cast<__ubuf__ bfloat16_t*>(d.GetPhyAddr());
            o1p = reinterpret_cast<__ubuf__ bfloat16_t*>(out1.GetPhyAddr());
        }
        if constexpr (Op == 2) wp = reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr());
        VectorTile<Op>(reinterpret_cast<__ubuf__ bfloat16_t*>(g.GetPhyAddr()), up, dp, wp,
            reinterpret_cast<__ubuf__ bfloat16_t*>(out0.GetPhyAddr()), o1p, nr, nc);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams store{nr, static_cast<uint32_t>(nc*sizeof(bfloat16_t)), 0,
            static_cast<uint32_t>((outStride-nc)*sizeof(bfloat16_t)), 0};
        const uint64_t outputOffset = static_cast<uint64_t>(r0)*outStride+c0;
        DataCopyPad(outGm[outputOffset], out0, store);
        if constexpr (Op == 1) DataCopyPad(outGm[outputOffset+width], out1, store);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
}

extern "C" int megamoe_train_pointwise(int op, void* input, void* da, void* p,
    void* output, uint32_t rows, uint32_t width, void* stream, uint32_t vector_cores) {
    if (op<0 || op>2 || !input || !output || !width || width%128 || !vector_cores ||
        width>(1U<<20) || (op==1 && !da) || (op==2 && !p)) return 2;
    if (!rows) return 0;
    const uint64_t tiles = ((static_cast<uint64_t>(rows)+kRows-1)/kRows)*((width+kCols-1)/kCols);
    const uint32_t blocks = tiles<vector_cores ? static_cast<uint32_t>(tiles) : vector_cores;
    if (op==0) Pointwise<0><<<blocks,0,stream>>>((GM_ADDR)input,(GM_ADDR)da,(GM_ADDR)p,(GM_ADDR)output,rows,width);
    if (op==1) Pointwise<1><<<blocks,0,stream>>>((GM_ADDR)input,(GM_ADDR)da,(GM_ADDR)p,(GM_ADDR)output,rows,width);
    if (op==2) Pointwise<2><<<blocks,0,stream>>>((GM_ADDR)input,(GM_ADDR)da,(GM_ADDR)p,(GM_ADDR)output,rows,width);
    return 0;
}
