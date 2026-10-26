// Experimental compute component of the BF16 training wave pipeline.
// One MIX launch performs grouped W1 GEMM and SwiGLU, retaining rounded GU.
// No SHMEM, second GEMM, gating or backward is fused in this component.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include <cstdint>
#include "basic_api/kernel_basic_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "platform/platform_ascendc.h"

namespace {
using namespace AscendC;
constexpr uint32_t kM = 128, kN = 256, kCores = 32, kFlagStride = 32;
constexpr uint32_t kExperts = 4, kVectorRows = 8;
using Arch = Catlass::Arch::Ascend950;
using Row = Catlass::layout::RowMajor;
using Col = Catlass::layout::ColumnMajor;
using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false>;
using L1 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, bfloat16_t, Row,
    bfloat16_t, Col, bfloat16_t, Row>;
using Block = Catlass::Gemm::Block::BlockMmadTla<Policy, L1, L0,
    bfloat16_t, bfloat16_t, bfloat16_t, void, Copy>;
struct Params {
    GM_ADDR x;
    GM_ADDR w;
    GM_ADDR gu;
    GM_ADDR a;
    GM_ADDR flags;
    uint32_t rows, h, i, counts[kExperts], rowBegin[kExperts], jobBegin[kExperts+1];
};
struct Job { uint32_t expert, localRow, globalRow, col, rows, cols; };

__aicore__ inline Job Locate(const Params& p, uint32_t job) {
    uint32_t e = 0;
    while (job >= p.jobBegin[e+1]) ++e; // host guarantees job < total
    const uint32_t local = job-p.jobBegin[e], nTiles=(p.i+kN-1)/kN;
    const uint32_t r = local/nTiles*kM, c = local%nTiles*kN;
    return {e,r,p.rowBegin[e]+r,c,
        p.counts[e]-r<kM ? p.counts[e]-r : kM, p.i-c<kN ? p.i-c : kN};
}

constexpr Reg::CastTrait toFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait toBf16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// Identical FP32 operation order and BF16 rounding to the qualified P1 Op0.
__aicore__ inline void Activate(__ubuf__ bfloat16_t* g, __ubuf__ bfloat16_t* u,
    __ubuf__ bfloat16_t* out, uint16_t rows, uint16_t cols) {
    constexpr uint16_t vlen = VECTOR_REG_WIDTH/sizeof(float);
    __VEC_SCOPE__ {
        Reg::RegTensor<bfloat16_t> gb, ub, result;
        Reg::RegTensor<float> gf, uf, neg, denominator, tmp, value;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        for (uint16_t r=0; r<rows; ++r) {
            for (uint16_t v=0; v<cols/vlen; ++v) {
                uint32_t off=r*cols+v*vlen;
                Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gb,g+off);
                Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(ub,u+off);
                Reg::Cast<float,bfloat16_t,toFloat>(gf,gb,mask);
                Reg::Cast<float,bfloat16_t,toFloat>(uf,ub,mask);
                Reg::Muls(neg,gf,-1.0F,mask);
                Reg::Exp(denominator,neg,mask);
                Reg::Adds(denominator,denominator,1.0F,mask);
                Reg::Div(tmp,gf,denominator,mask);
                Reg::Mul(value,tmp,uf,mask);
                Reg::Cast<bfloat16_t,float,toBf16>(result,value,mask);
                Reg::DataCopy<bfloat16_t,Reg::StoreDist::DIST_PACK_B32>(out+off,result,mask);
            }
        }
    }
}

__aicore__ inline void Cube(const Params& p) {
    Catlass::Arch::Resource<Arch> resource;
    Block mm(resource);
    GlobalTensor<bfloat16_t> gx,gw,ggu;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.x));
    gw.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.w));
    ggu.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.gu));
    auto tx=tla::MakeTensor(gx,tla::MakeLayout<bfloat16_t,Row>(p.rows,p.h),Catlass::Arch::PositionGM{});
    auto tc=tla::MakeTensor(ggu,tla::MakeLayout<bfloat16_t,Row>(p.rows,2*p.i),Catlass::Arch::PositionGM{});
    for (uint32_t j=GetBlockIdx(); j<p.jobBegin[kExperts]; j+=kCores) {
        const Job q=Locate(p,j);
        auto tw=tla::MakeTensor(gw[static_cast<uint64_t>(q.expert)*2*p.i*p.h],
            tla::MakeLayout<bfloat16_t,Col>(p.h,2*p.i),Catlass::Arch::PositionGM{});
        auto a=GetTile(tx,tla::MakeCoord(q.globalRow,0),tla::MakeShape(q.rows,p.h));
        for (uint32_t half=0; half<2; ++half) {
            const uint32_t n=q.col+half*p.i;
            auto b=GetTile(tw,tla::MakeCoord(0,n),tla::MakeShape(p.h,q.cols));
            auto c=GetTile(tc,tla::MakeCoord(q.globalRow,n),tla::MakeShape(q.rows,q.cols));
            mm(a,b,c,Catlass::GemmCoord{q.rows,q.cols,p.h});
        }
        // This policy is synchronous at block level. Drain both FIX stores
        // before publishing; neither GU nor this flag is reused in this call.
        static_assert(!Policy::ASYNC, "Async policy requires explicit block completion");
        SetFlag<HardEvent::FIX_S>(EVENT_ID0); WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(p.flags)+static_cast<uint64_t>(j)*kFlagStride,1);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Vector(const Params& p) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> gb,ub,ob;
    pipe.InitBuffer(gb,kVectorRows*kN*sizeof(bfloat16_t));
    pipe.InitBuffer(ub,kVectorRows*kN*sizeof(bfloat16_t));
    pipe.InitBuffer(ob,kVectorRows*kN*sizeof(bfloat16_t));
    auto g=gb.Get<bfloat16_t>(), u=ub.Get<bfloat16_t>(), out=ob.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gu,a;
    gu.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.gu));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.a));
    const uint32_t core=GetBlockIdx()/GetSubBlockNum(), half=GetSubBlockIdx();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (uint32_t j=core; j<p.jobBegin[kExperts]; j+=kCores) {
        const Job q=Locate(p,j);
        const uint32_t start=half*(kM/2), end=q.rows<(half+1)*(kM/2)?q.rows:(half+1)*(kM/2);
        if (start>=end) continue; // no wait for an empty consumer
        auto flag=reinterpret_cast<__gm__ int32_t*>(p.flags)+static_cast<uint64_t>(j)*kFlagStride;
        while (ReadGmByPassDCache(flag)!=1) {
            int64_t cycle=GetSystemCycle();
            while (GetSystemCycle()-cycle<64) {}
        }
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        for (uint32_t row=start; row<end; row+=kVectorRows) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            const uint16_t nr=end-row<kVectorRows?end-row:kVectorRows;
            uint64_t off=static_cast<uint64_t>(q.globalRow+row)*2*p.i+q.col;
            DataCopyExtParams load{nr,q.cols*2,(2*p.i-q.cols)*2,0,0};
            DataCopyPadExtParams<bfloat16_t> noPad{false,0,0,0};
            DataCopyPad(g,gu[off],load,noPad); DataCopyPad(u,gu[off+p.i],load,noPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Activate(reinterpret_cast<__ubuf__ bfloat16_t*>(g.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(u.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t*>(out.GetPhyAddr()),nr,q.cols);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(a[static_cast<uint64_t>(q.globalRow+row)*p.i+q.col],out,
                {nr,q.cols*2,0,(p.i-q.cols)*2,0});
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

[[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void UpSwiGLU(Params p) {
    if constexpr (g_coreType == AscendC::AIC) { Cube(p); }
    if constexpr (g_coreType == AscendC::AIV) { Vector(p); }
}
}

extern "C" int megamoe_train_up_swiglu(void* x,void* w,void* gu,void* a,void* flags,
    uint32_t rows,uint32_t h,uint32_t i,const uint32_t* counts,uint64_t flag_bytes,void* stream) {
    if (!counts || !h || !i || h%128 || i%128 || h>(1U<<20) || i>(1U<<20)) return 2;
    auto hw=platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!hw || hw->GetCoreNumAic()!=kCores || hw->GetCoreNumAiv()!=2*kCores) return 3;
    Params p{};p.x=(GM_ADDR)x;p.w=(GM_ADDR)w;p.gu=(GM_ADDR)gu;p.a=(GM_ADDR)a;p.flags=(GM_ADDR)flags;
    p.rows=rows;p.h=h;p.i=i;
    uint64_t r=0,j=0;
    for (uint32_t e=0;e<kExperts;++e) {
        p.counts[e]=counts[e];p.rowBegin[e]=static_cast<uint32_t>(r);p.jobBegin[e]=static_cast<uint32_t>(j);
        r+=counts[e];j+=((static_cast<uint64_t>(counts[e])+kM-1)/kM)*((i+kN-1)/kN);
        if (r>UINT32_MAX || j>UINT32_MAX) return 2;
    }
    p.jobBegin[kExperts]=static_cast<uint32_t>(j);
    if (r!=rows || flag_bytes<j*kFlagStride*sizeof(int32_t)) return 2;
    if (!rows) return 0;
    if (!x || !w || !gu || !a || !flags) return 2;
    UpSwiGLU<<<kCores,0,stream>>>(p);
    return 0;
}

extern "C" const char* megamoe_train_up_swiglu_backend() {
    return "experimental-CATLASS950-BF16-grouped-up-SwiGLU-v1";
}
