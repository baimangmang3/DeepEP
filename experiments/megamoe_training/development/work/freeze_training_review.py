import hashlib,json,pathlib,zipfile,sys
prefix=sys.argv[1] if len(sys.argv)>1 else 'training_review_source'
root=pathlib.Path('work/DeepEP/experiments/megamoe_training')
files=[]
for path in sorted(root.rglob('*')):
 if path.is_file() and not {'build','__pycache__'} & set(path.relative_to(root).parts):
  files.append(dict(path=path.relative_to(root).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
manifest=dict(schema=1,base_git='35fa7d91b321340c153b098bf166202645976a75',files=files,native_library_sha256='f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32')
pathlib.Path('outputs/'+prefix+'_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile('outputs/'+prefix+'.zip','w',zipfile.ZIP_DEFLATED) as z:
 for row in files:z.write(root/row['path'],row['path'])
 z.writestr('source-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'source_files':len(files),'zip_sha256':hashlib.sha256(pathlib.Path('outputs/'+prefix+'.zip').read_bytes()).hexdigest()}))
