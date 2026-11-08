# Shared helpers for cfi.ps1, cfi-doctor.ps1 and cfi-selftest.ps1 (dot-sourced; not run directly).
# Works in Windows PowerShell 5.1 and PowerShell 7.

$script:CfiVersionMarker = 'cfi-install-manifest-v1'

function Get-CfiRedactor {
    # Names that must never appear in output that may be pasted into an issue.
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($v in @($env:USERPROFILE, $env:USERNAME, $env:COMPUTERNAME, $env:USERDOMAIN, $env:HOMEPATH)) {
        if ($v -and $v.Length -ge 3) { $names.Add($v) }
    }
    return , $names
}

function Redact([string]$Text) {
    if ($null -eq $Text) { return '' }
    $t = $Text
    $names = Get-CfiRedactor
    # Longest first so the profile path is replaced before the bare user name.
    foreach ($n in ($names | Sort-Object Length -Descending)) {
        $t = [regex]::Replace($t, [regex]::Escape($n), '<redacted>', 'IgnoreCase')
    }
    $t = [regex]::Replace($t, '(?i)([A-Z]:\\Users\\)[^\\\s"'']+', '$1<user>')
    $t = [regex]::Replace($t, '\b(\d{1,3}\.){3}\d{1,3}\b', '<ip>')
    $t = [regex]::Replace($t, '[A-Za-z0-9._-]+@[A-Za-z0-9.-]+', '<email-or-host>')
    return $t
}

function Write-Safe([string]$Text) { Write-Host (Redact $Text) }

function Get-Sha256Hex([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Initialize-CfiNative {
    if ('CfiNative' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CfiNative {
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32", CharSet = CharSet.Ansi)]
    static extern IntPtr GetProcAddress(IntPtr module, string name);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate IntPtr InfoFn();
    public static string BuildInfo(string path) {
        IntPtr h = LoadLibraryW(path);
        if (h == IntPtr.Zero) return "load-failed(win32=" + Marshal.GetLastWin32Error() + ")";
        IntPtr p = GetProcAddress(h, "cfi_build_info");
        if (p == IntPtr.Zero) return "not-exported(older build)";
        InfoFn f = (InfoFn)Marshal.GetDelegateForFunctionPointer(p, typeof(InfoFn));
        return Marshal.PtrToStringAnsi(f());
    }
}
'@
}

function Get-ShimBuildInfo([string]$DllPath) {
    try { Initialize-CfiNative; return [CfiNative]::BuildInfo($DllPath) }
    catch { return 'unavailable' }
}

function Find-LevelZeroLoader {
    $candidates = @(
        (Join-Path $env:SystemRoot 'System32\ze_loader.dll'),
        (Join-Path $env:SystemRoot 'SysWOW64\ze_loader.dll')
    )
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c -PathType Leaf) { return $c } }
    $cmd = Get-Command 'ze_loader.dll' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-IntelGpus {
    $result = @()
    try {
        foreach ($g in (Get-CimInstance Win32_VideoController -ErrorAction Stop)) {
            $ven = ''; $dev = ''
            if ($g.PNPDeviceID -match 'VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})') { $ven = $Matches[1]; $dev = $Matches[2] }
            $result += [pscustomobject]@{
                Name = $g.Name; Driver = $g.DriverVersion; DriverDate = $g.DriverDate
                Vendor = $ven; Device = $dev; IsIntel = ($ven -eq '8086' -or $g.Name -match 'Intel')
            }
        }
    } catch { }
    return $result
}

function Test-VcRuntime {
    foreach ($k in @('HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
                     'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64')) {
        $p = Get-ItemProperty -Path $k -ErrorAction SilentlyContinue
        if ($p -and $p.Installed -eq 1) { return "$($p.Major).$($p.Minor).$($p.Bld)" }
    }
    return $null
}

function Get-OllamaVersion([string]$Exe) {
    if (-not $Exe) {
        $cmd = Get-Command 'ollama.exe' -ErrorAction SilentlyContinue
        if (-not $cmd) { return $null }
        $Exe = $cmd.Source
    }
    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { return $null }
    $v = (Get-Item -LiteralPath $Exe).VersionInfo.ProductVersion
    if ($v) { return $v.Trim() }
    return 'unknown'
}

function Get-CfiDefaultInstallDir {
    if ($env:LOCALAPPDATA) { return (Join-Path $env:LOCALAPPDATA 'cuda-for-intel\install') }
    return (Join-Path $env:TEMP 'cuda-for-intel\install')
}
