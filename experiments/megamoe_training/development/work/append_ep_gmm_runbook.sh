python3 - <<'PY'
from pathlib import Path
p=Path('/home/zzy/deploy/megamoe-20261024/training/RUNBOOK.md')
text=p.read_text();marker='## ep_gmm_fused 新增对照'
if marker not in text:
 p.write_text(text+'\n\n'+marker+'\n\n先读取同目录 EP_GMM_PREFLIGHT.md。当前 pinned 原版仅EP1-8、BF16且无autograd；64卡EP4多域说明不是单域EP16支持。原版EP8应保留独立wheel/环境/哈希作为对照，当前只有依赖准备，尚无该对照wheel或设备资格。EP16适配版必须另立版本和补丁身份，核查rank数组/metadata ABI/UB/workspace/queue-owner/ready与跨机UDMA连接；不能仅放宽Python上限。先完整精度，再相同scope性能及带宽/通算掩盖/MFU采证；打点诊断构建与无打点性能分开。\n\n03:00及每10分钟两机占用门禁、timeout和只清理本任务进程组规则继续适用；原版可先以两台各4卡构成单EP8域验收。不得将CCU改名AIV，不修改或重打包现有13文件训练快照及既有证据包。\n')
print('RUNBOOK_EP_GMM_NOTE_PRESENT',marker in p.read_text())
PY