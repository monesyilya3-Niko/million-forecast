# ── Windows 定时任务设置 ──
# 在服务器 PowerShell（管理员）中执行
# 每 6 小时自动更新数据

$PythonPath = "C:\Program Files\Python313\python.exe"
$ScriptPath = "C:\million-forecast\scripts\scheduled_update.py"
$TaskName = "MillionForecast-DataUpdate"

# 删除旧任务（如果存在）
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# 创建任务
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory "C:\million-forecast"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 365)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "百万竞猜 - 每6小时自动更新竞彩/阵容/赛果数据"

Write-Host "定时任务已创建: $TaskName"
Write-Host "执行频率: 每 6 小时"
Write-Host "脚本路径: $ScriptPath"
