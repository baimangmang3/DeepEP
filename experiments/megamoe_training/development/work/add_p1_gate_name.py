from pathlib import Path
p=Path('work/training_occupancy_gate.py');s=p.read_text().replace("'pta_initialization_probe.py'}", "'pta_initialization_probe.py','test_pointwise_npu.py','acl_probe_supervisor.py'}");p.write_text(s)
