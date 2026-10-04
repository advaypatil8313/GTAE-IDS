$wsh = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
Get-ChildItem -Path $desktop -Filter "*Run Project*.lnk" | ForEach-Object {
    $sc = $wsh.CreateShortcut($_.FullName)
    Write-Output "--- Shortcut: $($_.Name) ---"
    Write-Output "TargetPath: $($sc.TargetPath)"
    Write-Output "WorkingDirectory: $($sc.WorkingDirectory)"
    Write-Output "Arguments: $($sc.Arguments)"
    Write-Output "IconLocation: $($sc.IconLocation)"
}
