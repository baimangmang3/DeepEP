"""Local-only unique RUN preparation; unchanged reviewed controller is reused."""
import pathlib,ast
source=pathlib.Path('work/start_p2_ep16_serial_0557_once.py').read_text().replace('20261026_0557_p2_ep16_serial_once','20261026_0628_p2_ep16_serial_once')
needle="key=state.setdefault('p2_ep16_serial_hostname_control',{});assert not key.get('consumed_run_tag') and not key.get('permission_consumed')"
assert needle in source
source=source.replace(needle,needle+"\nassert key.get('pending_single_control') is True\nlatest_deadline=datetime.datetime.fromisoformat(key['ownership_recheck_not_before_UTC']);assert datetime.datetime.now(datetime.timezone.utc)>=latest_deadline,'Latest30min guard deadline not reached'")
ast.parse(source);pathlib.Path('work/start_p2_ep16_serial_0628_once.py').write_text(source)
print('Prepared only local RUN wrapper; controller unchanged, no execution.')
