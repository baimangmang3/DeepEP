$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path $PSScriptRoot -Parent
$sourceRoot = (Resolve-Path (Join-Path $workspaceRoot 'work/DeepEP/experiments/megamoe_training')).Path
$outputRoot = (Resolve-Path (Join-Path $workspaceRoot 'outputs')).Path
$zipPath = Join-Path $outputRoot 'training_source_snapshot.zip'
Add-Type -AssemblyName System.IO.Compression
$zipStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
$archive = [System.IO.Compression.ZipArchive]::new($zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $sourceFiles = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object { $_.FullName -notmatch '\\(__pycache__|build)\\' }
    foreach ($sourceFile in $sourceFiles) {
        $relativePath = $sourceFile.FullName.Substring($sourceRoot.Length + 1).Replace('\', '/')
        $entry = $archive.CreateEntry('experiments/megamoe_training/' + $relativePath)
        $entryStream = $entry.Open()
        $inputStream = [System.IO.File]::OpenRead($sourceFile.FullName)
        try { $inputStream.CopyTo($entryStream) } finally { $inputStream.Dispose(); $entryStream.Dispose() }
    }
} finally { $archive.Dispose(); $zipStream.Dispose() }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$manifest = @()
try {
    foreach ($entry in $archive.Entries) {
        $relativePath = $entry.FullName.Substring('experiments/megamoe_training/'.Length)
        $stream = $entry.Open()
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { $zipHash = [Convert]::ToHexString($sha.ComputeHash($stream)).ToLower() } finally { $stream.Dispose(); $sha.Dispose() }
        $fileHash = (Get-FileHash -LiteralPath (Join-Path $sourceRoot $relativePath) -Algorithm SHA256).Hash.ToLower()
        if ($fileHash -ne $zipHash) { throw "Archive mismatch: $relativePath" }
        $manifest += @{path=$entry.FullName; sha256=$fileHash; bytes=$entry.Length}
    }
} finally { $archive.Dispose() }
$manifestDocument = @{
    base_commit='35fa7d91b321340c153b098bf166202645976a75'
    catlass_commit='2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1'
    native_library_sha256='f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32'
    source_zip_sha256=(Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower()
    npu_tests_executed=$false
    files=$manifest
}
$manifestDocument | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outputRoot 'training_source_manifest.json') -Encoding utf8
@{verified_files=$manifest.Count;zip_sha256=$manifestDocument.source_zip_sha256} | ConvertTo-Json -Compress
