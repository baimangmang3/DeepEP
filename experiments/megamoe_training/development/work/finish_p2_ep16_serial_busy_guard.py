"""Local per-RUN EP16 guard bookkeeping, no remote or occupancy operation."""
import pathlib,sys
source=pathlib.Path('work/finish_p2_overlap_busy_guard.py').read_text()
source=source.replace('outputs/p2_ep2_overlap_hostname_preparation','outputs/p2_ep16_serial_hostname_preparation').replace('p2-ep2-bf16-overlap-controller','p2-ep16-bf16-serial-controller').replace('p2_ep2_overlap_hostname_control','p2_ep16_serial_hostname_control').replace('p2_overlap_pending_active_foreign_launcher_30min','p2_ep16_serial_pending_active_foreign_launcher_30min').replace('EP2 overlap','EP16 serial')
exec(compile(source,'EP16_serial_guard_bookkeeping','exec'))
