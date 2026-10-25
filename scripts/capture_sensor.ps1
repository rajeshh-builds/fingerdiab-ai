# capture_sensor.ps1
param([string]$message = "Scan your fingerprint for FingerDiab AI clinical analysis")

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($asyncOp, $type) {
    $asTask = $asTaskGeneric.MakeGenericMethod($type)
    $netTask = $asTask.Invoke($null, @($asyncOp))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

try {
    [Windows.Security.Credentials.UI.UserConsentVerifier,Windows.Security.Credentials.UI,ContentType=WindowsRuntime] | Out-Null
    $verifyOp = [Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync($message)
    $result = Await $verifyOp ([Windows.Security.Credentials.UI.UserConsentVerificationResult])
    Write-Output "WINRT_VERIFY_RESULT:$result"
} catch {
    Write-Output "WINRT_VERIFY_ERROR:$($_.Exception.Message)"
}
