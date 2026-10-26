"""Actual-map dev/inode/fstat-bound native hashing; stdlib only."""
import hashlib,os
from pathlib import Path
_HASH_CACHE={}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def mapped_libraries(maps_path=Path('/proc/self/maps')):
    """Record actual mappings; never search disk/install prefixes for missing SOs."""
    result = {'maps_path': str(maps_path), 'libraries': [], 'errors': []}
    try:
        lines = maps_path.read_text().splitlines()
    except OSError as exc:
        result['errors'].append(repr(exc))
        return result
    mappings = {}
    blocked = set()
    for line_number, line in enumerate(lines, 1):
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith('/'):
            continue
        raw = fields[5].replace('\\040', ' ')
        name = Path(raw).name.lower()
        if '.so' not in name or not (any((v in name for v in ('hccl', 'hcomm', 'runtime', 'ascendcl', 'opapi', 'shmem', 'kernels', 'torch_npu', 'megamoe'))) or any(v in raw for v in ('/torch_npu/', '/deep_ep/', '/ep_gmm_fused/'))):
            continue
        if raw.endswith(' (deleted)'):
            result['errors'].append('Deleted actual mapping: ' + raw)
            continue
        try:
            major, minor = (int(v, 16) for v in fields[3].split(':'))
            inode = int(fields[4])
            if major < 0 or minor < 0:
                raise ValueError('Mapped device major/minor must be nonnegative')
            if inode <= 0:
                raise ValueError('Mapped inode must be positive')
            mappings.setdefault(raw, []).append(dict(line=line_number, address_range=fields[0], permissions=fields[1], offset=fields[2], device=fields[3], device_major=major, device_minor=minor, inode=inode))
        except (ValueError, TypeError) as exc:
            blocked.add(raw)
            result['errors'].append(f'Invalid maps identity {raw}: {exc!r}')
    for raw, entries in sorted(mappings.items()):
        if raw in blocked:
            continue
        identities = {(row['device_major'], row['device_minor'], row['inode']) for row in entries}
        if len(identities) != 1:
            result['errors'].append('Same mapped path has inconsistent dev/inode: ' + raw)
            continue
        major, minor, inode = next(iter(identities))
        path = Path(raw)

        def signature(st):
            return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        try:
            if not hasattr(os, 'major') or not hasattr(os, 'minor'):
                raise OSError('POSIX dev major/minor decoder unavailable; cannot bind maps identity')
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                if (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino) != (major, minor, inode):
                    raise OSError('Opened file fstat dev/inode does not match mapped object')
                if before.st_size <= 0:
                    raise OSError('Mapped library is not a nonempty readable file')
                key = (major, minor, inode, *signature(before), before.st_ctime_ns)
                digest = _HASH_CACHE.get(key)
                if digest is None:
                    h = hashlib.sha256()
                    for block in iter(lambda: stream.read(8 << 20), b''):
                        h.update(block)
                    digest = h.hexdigest()
                after = os.fstat(stream.fileno())
                resolved = path.resolve(strict=True)
                if signature(before) != signature(after) or before.st_ctime_ns != after.st_ctime_ns or signature(path.stat()) != signature(after) or (signature(resolved.stat()) != signature(after)):
                    raise OSError('Mapped library descriptor/path changed during hashing')
                _HASH_CACHE[key] = digest
            result['libraries'].append(dict(path=str(resolved), mapped_path=raw, sha256=digest, bytes=after.st_size, device=after.st_dev, inode=after.st_ino, map_device_major=major, map_device_minor=minor, map_inode=inode, map_entries=entries))
        except (OSError, ValueError) as exc:
            result['errors'].append(f'Mapped file verification {raw}: {exc!r}')
    names = [Path(item['path']).name for item in result['libraries']]
    result['present'] = {role: any((check(name) for name in names)) for role, check in {'hccl_or_hcomm': lambda n: 'hccl' in n or 'hcomm' in n, 'runtime': lambda n: 'runtime' in n, 'ascendcl': lambda n: 'ascendcl' in n, 'torch_npu': lambda n: 'torch_npu' in n or any(('/torch_npu/' in p['path'] for p in result['libraries']))}.items()}
    result['missing_roles_are_diagnostic'] = 'Lazy/unmapped is recorded, not guessed or explicitly CDLL-loaded; this is not a qualification identity.'
    return result
