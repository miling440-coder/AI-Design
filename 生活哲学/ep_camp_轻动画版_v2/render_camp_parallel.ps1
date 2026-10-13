# 露营动画：在 Windows PowerShell 5.1 / PowerShell 7 下分段并行 CUDA 渲染
# 用法：
#   .\render_camp_parallel.ps1 -MaxParallel 2
#   .\render_camp_parallel.ps1 -MaxParallel 1 -Scene scene_camp_optimized.py
#   .\render_camp_parallel.ps1 -MaxParallel 2 -Merge
#   .\render_camp_parallel.ps1 -MaxParallel 2 -Merge -AddAudio
param(
    [ValidateRange(1, 9)][int]$MaxParallel = 2,
    [ValidateSet('scene_camp.py', 'scene_camp_optimized.py')][string]$Scene = 'scene_camp.py',
    [switch]$Merge,
    [switch]$AddAudio
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if ($AddAudio -and -not $Merge) {
    throw '使用 -AddAudio 时，请同时指定 -Merge。'
}
if (-not (Test-Path -LiteralPath $Scene -PathType Leaf)) {
    throw "场景文件不存在：$Scene"
}

$pythonCommand = Get-Command python -ErrorAction Stop
$pythonExe = $pythonCommand.Source
$logDir = Join-Path $PSScriptRoot 'render_logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# range() 的结束帧不包含在输出中；121.9s * 30fps = 3657 帧。
$tasks = @(
    @{ Start = 0;    End = 438;  Name = 's1.mp4' },
    @{ Start = 438;  End = 747;  Name = 's2.mp4' },
    @{ Start = 747;  End = 975;  Name = 's3.mp4' },
    @{ Start = 975;  End = 1171; Name = 's4.mp4' },
    @{ Start = 1171; End = 1582; Name = 's5.mp4' },
    @{ Start = 1582; End = 2164; Name = 's6.mp4' },
    @{ Start = 2164; End = 2815; Name = 's7.mp4' },
    @{ Start = 2815; End = 3217; Name = 's8.mp4' },
    @{ Start = 3217; End = 3657; Name = 's9.mp4' }
)

$active = New-Object 'System.Collections.Generic.List[object]'
$failures = New-Object 'System.Collections.Generic.List[string]'

function Finish-Completed {
    param([System.Collections.Generic.List[object]]$Active, [System.Collections.Generic.List[string]]$Failures)
    foreach ($item in @($Active.ToArray())) {
        if (-not $item.Process.HasExited) { continue }
        $item.Process.WaitForExit()
        $exitCode = $item.Process.ExitCode
        if ($exitCode -eq 0) {
            Write-Host "[完成] $($item.Name)"
        } else {
            $Failures.Add($item.Name)
            Write-Warning "[失败] $($item.Name)，退出码 $exitCode；请检查 $($item.Log)"
        }
        [void]$Active.Remove($item)
        $item.Process.Dispose()
    }
}

Write-Host "工作目录：$PSScriptRoot"
Write-Host "场景：$Scene | 同时运行的 Python 进程数：$MaxParallel"
Write-Host 'CUDA 每个进程会各占一份显存；如遇 OOM，请改用 -MaxParallel 1。'

foreach ($task in $tasks) {
    while ($active.Count -ge $MaxParallel) {
        Finish-Completed -Active $active -Failures $failures
        if ($active.Count -ge $MaxParallel) { Start-Sleep -Seconds 2 }
    }

    $name = $task.Name
    $stem = [IO.Path]::GetFileNameWithoutExtension($name)
    $stdout = Join-Path $logDir "$stem.stdout.log"
    $stderr = Join-Path $logDir "$stem.stderr.log"
    $argsList = @('-u', $Scene, 'seg', [string]$task.Start, [string]$task.End, $name)
    Write-Host "[启动] $name : $($task.Start) -> $($task.End)"
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $argsList `
        -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $active.Add([pscustomobject]@{ Name = $name; Process = $proc; Log = $stderr })
}

while ($active.Count -gt 0) {
    Finish-Completed -Active $active -Failures $failures
    if ($active.Count -gt 0) { Start-Sleep -Seconds 2 }
}

if ($failures.Count -gt 0) {
    throw "以下片段渲染失败：$($failures -join ', ')。未执行拼接。"
}
Write-Host '[完成] 9 段全部渲染成功。'

if ($Merge) {
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        throw '找不到 ffmpeg；请安装并加入 PATH 后重试拼接。'
    }
    # 拼接清单仅使用 ASCII 文件名，避免 Windows PowerShell 5.1 UTF-8 BOM 兼容问题。
    $lines = $tasks | ForEach-Object { "file '$($_.Name)'" }
    Set-Content -LiteralPath 'camp_segments.txt' -Value $lines -Encoding Ascii
    Write-Host '[拼接] camp_segments_video.mp4'
    & ffmpeg -y -f concat -safe 0 -i 'camp_segments.txt' -c copy 'camp_segments_video.mp4'
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg 拼接失败，退出码 $LASTEXITCODE" }
    if ($AddAudio) {
        Write-Host '[音频] 运行原有脚本的声音合成与混音流程…'
        & $pythonExe $Scene audio 'camp_segments_video.mp4'
        if ($LASTEXITCODE -ne 0) { throw "音频处理失败，退出码 $LASTEXITCODE" }
    }
}
