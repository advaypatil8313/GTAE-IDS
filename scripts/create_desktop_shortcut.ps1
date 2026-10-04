$desktopPath = [Environment]::GetFolderPath('Desktop')
$wsh = New-Object -ComObject WScript.Shell

$targetBatch = "C:\Users\Aniket Patil\Desktop\GTAE-IDS\scripts\launch_gtae_ids.bat"
$workingDir = "C:\Users\Aniket Patil\Desktop\GTAE-IDS"
$icon = "$env:SystemRoot\System32\shell32.dll,14"

# Remove any old / corrupted variations
Get-ChildItem -Path $desktopPath -Filter "*Run Project*.lnk" | Remove-Item -Force

# 1. En-dash shortcut: "GTAE-IDS – Run Project.lnk"
$dashChar = [char]0x2013
$fileName1 = "GTAE-IDS " + $dashChar + " Run Project.lnk"
$path1 = Join-Path $desktopPath $fileName1

$sc1 = $wsh.CreateShortcut($path1)
$sc1.TargetPath = $targetBatch
$sc1.WorkingDirectory = $workingDir
$sc1.Description = "One-click launcher for GTAE-IDS Review-II Demo and Web Frontend"
$sc1.IconLocation = $icon
$sc1.Save()

# 2. Hyphen shortcut: "GTAE-IDS - Run Project.lnk"
$fileName2 = "GTAE-IDS - Run Project.lnk"
$path2 = Join-Path $desktopPath $fileName2

$sc2 = $wsh.CreateShortcut($path2)
$sc2.TargetPath = $targetBatch
$sc2.WorkingDirectory = $workingDir
$sc2.Description = "One-click launcher for GTAE-IDS Review-II Demo and Web Frontend"
$sc2.IconLocation = $icon
$sc2.Save()

Write-Output "Shortcuts created successfully:"
Write-Output "1: $path1"
Write-Output "2: $path2"
