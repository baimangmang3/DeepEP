from pathlib import Path
p=Path('work/run_training_perf_checked.sh');s=p.read_text();s=s.replace('source "$T/activate-qualified-p0.sh"','source "$T/activate-qualified-p0.sh"\nexport CANDIDATE_SOURCE_ROOT="$T/candidate-p1-training/p1_pointwise"\nexport POINTWISE_SO="$T/candidate-p1/build/libmegamoe_pointwise.so"');Path('work/run_training_p1_checked.sh').write_bytes(s.encode())
p=Path('work/training_occupancy_gate.py');s=p.read_text().replace("'test_pointwise_npu.py'", "'test_pointwise_npu.py','validate_training_p1.py'");p.write_text(s)
