// P4 compute-only512-row wave: canonical BF16 GU/A/Z saves.
// No SHMEM, gating, router, backward or complete training qualification.
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
constexpr uint32_t kExperts = 4, kVectorRows = 8, kWave = 512;
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
    GM_ADDR x; GM_ADDR w1; GM_ADDR w2; GM_ADDR gu;
    GM_ADDR a; GM_ADDR z; GM_ADDR flags;
    uint32_t rows,h,i,counts[kExperts],rowBegin[kExperts],jobBegin[kExperts+1];
};
struct Pair { uint32_t row,col,rows,cols; };
__aicore__ inline uint32_t Min(uint32_t a,uint32_t b) { return a<b?a:b; }
__aicore__ inline Pair LocatePair(const Params& p,uint32_t expert,uint64_t wave,uint32_t j) {
    const uint32_t nt=(p.i+kN-1)/kN, mr=j/nt*kM;
    const uint32_t wr=Min(kWave,static_cast<uint32_t>(p.counts[expert]-wave));
    return {static_cast<uint32_t>(p.rowBegin[expert]+wave+mr),j%nt*kN,
        Min(kM,wr-mr),Min(kN,p.i-j%nt*kN)};
}
__aicore__ inline __gm__ int32_t* Ready(const Params& p,uint32_t channel,uint32_t job) {
    return reinterpret_cast<__gm__ int32_t*>(p.flags)+
        (static_cast<uint64_t>(channel)*p.jobBegin[kExperts]+job)*kFlagStride;
}
__aicore__ inline void WaitReady(__gm__ int32_t* flag) {
    while (ReadGmByPassDCache(flag)!=1) {
        int64_t cycle=GetSystemCycle();
        while (GetSystemCycle()-cycle<64) {}
    }
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
    GlobalTensor<bfloat16_t> gx,gw1,gw2,ggu,ga,gz;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.x));
    gw1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.w1));
    gw2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.w2));
    ggu.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.gu));
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.a));
    gz.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.z));
    auto tx=tla::MakeTensor(gx,tla::MakeLayout<bfloat16_t,Row>(p.rows,p.h),Catlass::Arch::PositionGM{});
    auto tgu=tla::MakeTensor(ggu,tla::MakeLayout<bfloat16_t,Row>(p.rows,2*p.i),Catlass::Arch::PositionGM{});
    auto ta=tla::MakeTensor(ga,tla::MakeLayout<bfloat16_t,Row>(p.rows,p.i),Catlass::Arch::PositionGM{});
    auto tz=tla::MakeTensor(gz,tla::MakeLayout<bfloat16_t,Row>(p.rows,p.h),Catlass::Arch::PositionGM{});
    const uint32_t core=GetBlockIdx(),ni=(p.i+kN-1)/kN,nh=(p.h+kN-1)/kN;
    for (uint32_t e=0;e<kExperts;++e) {
        auto tw1=tla::MakeTensor(gw1[static_cast<uint64_t>(e)*2*p.i*p.h],
            tla::MakeLayout<bfloat16_t,Col>(p.h,2*p.i),Catlass::Arch::PositionGM{});
        // Canonical W2[e,H,I] bytes are column-major [I,H], with no packing.
        auto tw2=tla::MakeTensor(gw2[static_cast<uint64_t>(e)*p.h*p.i],
            tla::MakeLayout<bfloat16_t,Col>(p.i,p.h),Catlass::Arch::PositionGM{});
        for (uint64_t wave=0;wave<p.counts[e];wave+=kWave) {
            const uint32_t wr=Min(kWave,static_cast<uint32_t>(p.counts[e]-wave));
            const uint32_t mt=(wr+kM-1)/kM,pairs=mt*ni;
            const uint32_t base=p.jobBegin[e]+static_cast<uint32_t>(wave/kM)*ni;
            // Every core finishes ALL its up pairs before ANY activation wait.
            for (uint32_t j=core;j<pairs;j+=kCores) {
                const Pair q=LocatePair(p,e,wave,j);
                auto x=GetTile(tx,tla::MakeCoord(q.row,0),tla::MakeShape(q.rows,p.h));
                for (uint32_t half=0;half<2;++half) {
                    const uint32_t n=q.col+half*p.i;
                    auto w=GetTile(tw1,tla::MakeCoord(0,n),tla::MakeShape(p.h,q.cols));
                    auto c=GetTile(tgu,tla::MakeCoord(q.row,n),tla::MakeShape(q.rows,q.cols));
                    mm(x,w,c,Catlass::GemmCoord{q.rows,q.cols,p.h});
                }
                static_assert(!Policy::ASYNC,"Explicit block completion required for async policy");
                SetFlag<HardEvent::FIX_S>(EVENT_ID0); WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
                WriteGmByPassDCache(Ready(p,0,base+j),1);
            }
            for (uint32_t t=core;t<mt*nh;t+=kCores) {
                const uint32_t m=t/nh,n=t%nh*kN;
                // Exact actual I tiles for this M tile; no padded producer waits.
                for (uint32_t tile=0;tile<ni;++tile) WaitReady(Ready(p,1,base+m*ni+tile));
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                const uint32_t nr=Min(kM,wr-m*kM),nc=Min(kN,p.h-n);
                const uint32_t row=static_cast<uint32_t>(p.rowBegin[e]+wave+m*kM);
                auto a=GetTile(ta,tla::MakeCoord(row,0),tla::MakeShape(nr,p.i));
                auto w=GetTile(tw2,tla::MakeCoord(0,n),tla::MakeShape(p.i,nc));
                auto z=GetTile(tz,tla::MakeCoord(row,n),tla::MakeShape(nr,nc));
                mm(a,w,z,Catlass::GemmCoord{nr,nc,p.i});
                SetFlag<HardEvent::FIX_S>(EVENT_ID0); WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Vector(const Params& p) {
    // AIV1 is deliberately idle: no SHMEM role or half-row activation here.
    if (GetSubBlockIdx()!=0) return;
    const uint32_t core=GetBlockIdx()/GetSubBlockNum(),ni=(p.i+kN-1)/kN;
    TPipe pipe;TBuf<TPosition::VECCALC> gb,ub,ob;
    pipe.InitBuffer(gb,kVectorRows*kN*sizeof(bfloat16_t));
    pipe.InitBuffer(ub,kVectorRows*kN*sizeof(bfloat16_t));
    pipe.InitBuffer(ob,kVectorRows*kN*sizeof(bfloat16_t));
    auto g=gb.Get<bfloat16_t>(),u=ub.Get<bfloat16_t>(),out=ob.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gu,a;
    gu.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.gu));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.a));
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (uint32_t e=0;e<kExperts;++e) {
        for (uint64_t wave=0;wave<p.counts[e];wave+=kWave) {
            const uint32_t wr=Min(kWave,static_cast<uint32_t>(p.counts[e]-wave));
            const uint32_t pairs=((wr+kM-1)/kM)*ni;
            const uint32_t base=p.jobBegin[e]+static_cast<uint32_t>(wave/kM)*ni;
            for (uint32_t j=core;j<pairs;j+=kCores) {
                const Pair q=LocatePair(p,e,wave,j);
                WaitReady(Ready(p,0,base+j));
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                // AIV0 covers the entire actual [0,q.rows), including rows64..127.
                for (uint32_t row=0;row<q.rows;row+=kVectorRows) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    const uint16_t nr=Min(kVectorRows,q.rows-row);
                    uint64_t off=static_cast<uint64_t>(q.row+row)*2*p.i+q.col;
                    DataCopyExtParams load{nr,q.cols*2,(2*p.i-q.cols)*2,0,0};
                    DataCopyPadExtParams<bfloat16_t> noPad{false,0,0,0};
                    DataCopyPad(g,gu[off],load,noPad);DataCopyPad(u,gu[off+p.i],load,noPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Activate(reinterpret_cast<__ubuf__ bfloat16_t*>(g.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ bfloat16_t*>(u.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ bfloat16_t*>(out.GetPhyAddr()),nr,q.cols);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(a[static_cast<uint64_t>(q.row+row)*p.i+q.col],out,
                        {nr,q.cols*2,0,(p.i-q.cols)*2,0});
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                // Visibility fence before scalar activation-ready publication.
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WriteGmByPassDCache(Ready(p,1,base+j),1);
            }
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

[[bisheng::core_ratio(1,2)]] __global__ __aicore__ void WaveCompute(Params p) {
    if constexpr(g_coreType==AscendC::AIC) { Cube(p); }
    if constexpr(g_coreType==AscendC::AIV) { Vector(p); }
}
}

extern "C" int megamoe_train_wave_compute(void* x,void* w1,void* w2,void* gu,void* a,void* z,void* flags,
    uint32_t rows,uint32_t h,uint32_t i,const uint32_t* counts,uint64_t flag_bytes,void* stream) {
    if (!counts || !h || !i || h%128 || i%128 || h>(1U<<20) || i>(1U<<20)) return 2;
    auto hw=platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!hw || hw->GetCoreNumAic()!=kCores || hw->GetCoreNumAiv()!=2*kCores) return 3;
    Params p{};p.x=(GM_ADDR)x;p.w1=(GM_ADDR)w1;p.w2=(GM_ADDR)w2;p.gu=(GM_ADDR)gu;
    p.a=(GM_ADDR)a;p.z=(GM_ADDR)z;p.flags=(GM_ADDR)flags;p.rows=rows;p.h=h;p.i=i;
    uint64_t r=0,j=0;
    for (uint32_t e=0;e<kExperts;++e) {
        p.counts[e]=counts[e];p.rowBegin[e]=static_cast<uint32_t>(r);p.jobBegin[e]=static_cast<uint32_t>(j);
        r+=counts[e];j+=((static_cast<uint64_t>(counts[e])+kM-1)/kM)*((i+kN-1)/kN);
        if (r>UINT32_MAX || j>UINT32_MAX) return 2;
    }
    p.jobBegin[kExperts]=static_cast<uint32_t>(j);
    if (r!=rows || flag_bytes<2*j*kFlagStride*sizeof(int32_t)) return 2;
    if (!rows) return 0;
    if (!x || !w1 || !w2 || !gu || !a || !z || !flags || !stream) return 2;
    WaveCompute<<<kCores,0,stream>>>(p);return 0;
}
extern "C" const char* megamoe_train_wave_compute_backend() {
    return "experimental-CATLASS950-BF16-512wave-GU-A-Z-v1";
}
