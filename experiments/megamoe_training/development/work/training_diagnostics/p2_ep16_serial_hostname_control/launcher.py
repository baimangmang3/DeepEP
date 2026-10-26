"""Per-RUN torchrun wrapper: event hostname is cached socket.gethostname().

Invocation arguments after this file are the original torchrun arguments.
No resolver/socket/environment/installed-file edits; rendezvous is unchanged.
This removes only the pinned agent event-helper FQDN lookup, not other DNS.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys

API='torch.distributed.elastic.agent.server.api'
API_SHA256='8184afa022897c622de9ff1b76504e2210550fa772ccbc2fba44279c913bc1fe'
_ACTIVE=False


def verify_source(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=API_SHA256:
        raise RuntimeError('Installed elastic agent API raw SHA differs from reviewed pin')
    tree=ast.parse(raw.decode('utf8'));parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    refs=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='_get_fq_hostname']
    if len(refs)!=1:raise RuntimeError('Unexpected event hostname helper references')
    ref=refs[0];call=parents[ref];mapping=parents[call]
    if (not isinstance(call,ast.Call) or call.args or call.keywords or not isinstance(mapping,ast.Dict) or
        not any(isinstance(k,ast.Constant) and k.value=='hostname' and v is call for k,v in zip(mapping.keys,mapping.values))):
        raise RuntimeError('Hostname helper is not confined to event hostname metadata')
    owner=ref
    while owner in parents and not isinstance(owner,ast.FunctionDef):owner=parents[owner]
    if not isinstance(owner,ast.FunctionDef) or owner.name!='_construct_event':raise RuntimeError('Unexpected hostname helper owner')
    return dict(helper_line=next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_fq_hostname'),
                call_line=ref.lineno,owner=owner.name,metadata_key='hostname')


def main(*,importer=importlib.import_module,gethostname=socket.gethostname):
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('Only one event hostname override per active launcher')
    hostname=gethostname()  # Cache before any Torch import; OS hostname, no FQDN lookup.
    if not isinstance(hostname,str) or not hostname or '\0' in hostname:raise ValueError('Invalid local hostname metadata')
    api=importer(API);path=Path(api.__file__).resolve(strict=True);proof=verify_source(path)
    original=api._get_fq_hostname
    if (not hasattr(original,'__code__') or Path(original.__code__.co_filename).resolve(strict=True)!=path or
        original.__code__.co_firstlineno!=proof['helper_line'] or original.__module__!=API):
        raise RuntimeError('Agent hostname helper already replaced or escaped pinned source')
    def cached_hostname():return hostname
    _ACTIVE=True;api._get_fq_hostname=cached_hostname
    try:
        print(json.dumps(dict(mode='elastic-agent-event-hostname=gethostname-cached',hostname=hostname,
            api_sha256=API_SHA256,api_path=str(path),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_proof=proof,scope='event metadata only; original torchrun argv/rendezvous/worker path retained')),file=sys.stderr,flush=True)
        run=importer('torch.distributed.run')
        return run.main()  # Original parser reads original sys.argv[1:]; no worker wrapper.
    finally:
        if api._get_fq_hostname is cached_hostname:api._get_fq_hostname=original
        _ACTIVE=False


if __name__=='__main__':sys.exit(main())
