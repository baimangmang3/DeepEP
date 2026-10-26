// Standalone CPU-only headers, no Torch/CANN/SHMEM. Not yet compiled locally.
// c++ -std=c++17 validate_native_cpu.cpp -o /ABS/native_cpu && /ABS/native_cpu
#include <cassert>
#include "candidate/experiments/ep_gmm_fused/csrc/tiling_validation.hpp"
#include "candidate/experiments/ep_gmm_fused/csrc/catccos_overlay/include/catccos/dgemm/moe_route_metadata.hpp"

int main() {
    namespace V=deepep::ep_gmm_fused::TilingValidation;
    constexpr uint64_t ub=248ULL*1024;
    V::AllToAllV up{128,256,256,64,1024,2,32,1,0,1};
    assert(!V::ValidateAllToAllV(up,65536,6144,16,32,ub));
    assert(!V::ValidateAllToAllV(up,65536,6400,16,32,ub));
    assert(V::ValidateAllToAllV(up,65536,6401,16,32,ub));
    assert(V::CheckHardware(17,32,ub));
    assert(V::CheckHardware(16,15,ub));
    assert(V::LocalCopyRowCapacity(6144,ub)==10);
    V::GroupedMatMul down{128,256,256,12,10,128,2,1};
    assert(!V::ValidateGroupedMatMul(down,6144,16,32,ub));
    down.localCopyTileRows=11;assert(V::ValidateGroupedMatMul(down,6144,16,32,ub));
    down.localCopyTileRows=32;assert(V::ValidateGroupedMatMul(down,6144,16,32,ub));
    down.localCopyTileRows=1;
    assert(!V::ValidateGroupedMatMul(down,51200,16,32,ub));
    assert(V::ValidateGroupedMatMul(down,51201,16,32,ub));
    down.localCopyTileRows=10;down.stageRows=0;down.commInterval=10;
    assert(V::ValidateGroupedMatMul(down,6144,16,32,ub));
    down.commInterval=12;assert(!V::ValidateGroupedMatMul(down,6144,16,32,ub));
    assert(sizeof(Catccos::DGemm::MoeRouteMetadata)==25232);
}
