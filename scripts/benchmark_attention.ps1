param(
    [string] $Compiler = "clang++",
    [int] $Tokens = 16,
    [int] $Width = 32,
    [int] $Iterations = 100,
    [int] $Repetitions = 7
)

$ErrorActionPreference = "Stop"
if ($Tokens -lt 2 -or $Width -lt 2 -or $Iterations -lt 1 -or $Repetitions -lt 3) {
    throw "Use Tokens >= 2, Width >= 2, Iterations >= 1, and Repetitions >= 3"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build\attention-benchmark"
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
$compilerCommand = Get-Command $Compiler -ErrorAction Stop
$compilerPath = $compilerCommand.Source
$compilerDirectory = Split-Path -Parent $compilerPath
$env:PATH = "$compilerDirectory;$env:PATH"
$proofline = Join-Path $buildDirectory "proofline.exe"
$compilerSources = @("main.cpp", "attention_cache.cpp", "model.cpp", "certificate.cpp", "codegen.cpp") |
    ForEach-Object { Join-Path $projectRoot "src\$_" }

& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 @compilerSources -o $proofline
if ($LASTEXITCODE -ne 0) { throw "Proofline build failed" }

$modelLines = [System.Collections.Generic.List[string]]::new()
$modelLines.Add("input2d $Tokens $Width f64")
for ($projection = 0; $projection -lt 3; $projection++) {
    $weights = [System.Collections.Generic.List[string]]::new()
    $biases = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt ($Width * $Width); $index++) {
        $raw = (($index * 13 + $projection * 17) % 61) - 30
        $weights.Add(([double]$raw / 1024.0).ToString(
            "R", [Globalization.CultureInfo]::InvariantCulture))
    }
    for ($index = 0; $index -lt $Width; $index++) {
        $raw = (($index * 7 + $projection * 3) % 19) - 9
        $biases.Add(([double]$raw / 512.0).ToString(
            "R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $line = "linear $Width f64 " + ($weights -join " ") + " " + ($biases -join " ")
    if ($projection -eq 0) {
        $modelLines.Add($line)
    } else {
        $modelLines.Add("linear_from input " + $line.Substring("linear ".Length))
    }
}
$modelLines.Add("attention v1 v2 v3 causal")
$modelPath = Join-Path $buildDirectory "attention.proof"
[System.IO.File]::WriteAllLines($modelPath, $modelLines,
    [System.Text.UTF8Encoding]::new($false))

$sampleInputs = [System.Collections.Generic.List[string]]::new()
for ($index = 0; $index -lt ($Tokens * $Width); $index++) {
    $raw = (($index * 11) % 43) - 21
    $sampleInputs.Add(([double]$raw / 64.0).ToString(
        "R", [Globalization.CultureInfo]::InvariantCulture))
}

function ConvertTo-NumericValues([string] $Text) {
    $tokens = $Text.Trim() -split '[,\s]+' | Where-Object { $_ }
    return @($tokens | ForEach-Object {
        [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture)
    })
}

function Get-Median([double[]] $Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count % 2 -eq 1) { return $sorted[[int]($sorted.Count / 2)] }
    return ($sorted[($sorted.Count / 2) - 1] + $sorted[$sorted.Count / 2]) / 2.0
}

$results = [System.Collections.Generic.List[object]]::new()
$fullPrecisionArguments = @("run", $modelPath) + $sampleInputs.ToArray()
$fullPrecisionOutput = & $proofline @fullPrecisionArguments
if ($LASTEXITCODE -ne 0) { throw "Full precision attention reference failed" }
$fullPrecisionValues = ConvertTo-NumericValues (($fullPrecisionOutput -split ": ")[-1])
foreach ($variant in @(
    @{ Name = "f64"; Flag = @() },
    @{ Name = "bf16"; Flag = @("--bf16") }
)) {
    $source = Join-Path $buildDirectory ($variant.Name + ".cpp")
    $executable = Join-Path $buildDirectory ($variant.Name + ".exe")
    & $proofline compile $modelPath $source @($variant.Flag)
    if ($LASTEXITCODE -ne 0) { throw "Code generation failed for $($variant.Name)" }
    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $source -o $executable
    if ($LASTEXITCODE -ne 0) { throw "Generated attention build failed for $($variant.Name)" }

    $runArguments = @("--bench", "$Iterations") + $sampleInputs.ToArray()
    $warmup = & $executable "--bench" "10" @($sampleInputs.ToArray())
    if ($LASTEXITCODE -ne 0) { throw "Attention warmup failed for $($variant.Name)" }
    $timings = [System.Collections.Generic.List[double]]::new()
    for ($repeat = 0; $repeat -lt $Repetitions; $repeat++) {
        $timing = & $executable @runArguments
        if ($LASTEXITCODE -ne 0) { throw "Attention timing failed for $($variant.Name)" }
        $match = [regex]::Match(($timing -join " "), "ns_per_inference=([0-9.eE+-]+)")
        if (-not $match.Success) { throw "Attention timing output was not recognized" }
        $timings.Add([double]::Parse($match.Groups[1].Value,
            [Globalization.CultureInfo]::InvariantCulture))
    }

    $referenceArguments = @("run", $modelPath)
    if ($variant.Name -eq "bf16") { $referenceArguments += "--bf16" }
    $referenceArguments += $sampleInputs.ToArray()
    $reference = & $proofline @referenceArguments
    if ($LASTEXITCODE -ne 0) { throw "Attention reference failed for $($variant.Name)" }
    $generated = & $executable @($sampleInputs.ToArray())
    if ($LASTEXITCODE -ne 0) { throw "Attention output run failed for $($variant.Name)" }
    $referenceValues = ConvertTo-NumericValues (($reference -split ": ")[-1])
    $generatedValues = ConvertTo-NumericValues ($generated -join " ")
    $maximumCodegenMismatch = 0.0
    $maximumF64Drift = 0.0
    for ($index = 0; $index -lt $referenceValues.Count; $index++) {
        $maximumCodegenMismatch = [Math]::Max($maximumCodegenMismatch,
            [Math]::Abs($referenceValues[$index] - $generatedValues[$index]))
        $maximumF64Drift = [Math]::Max($maximumF64Drift,
            [Math]::Abs($fullPrecisionValues[$index] - $generatedValues[$index]))
    }

    $parameterElementBytes = if ($variant.Name -eq "bf16") { 2 } else { 8 }
    $parameterBytes = 3 * ($Width * $Width + $Width) * $parameterElementBytes
    $activationBytes = 3 * $Tokens * $Width * 8 +
        3 * $Tokens * $Width * $parameterElementBytes + $Tokens * 8
    $fixedArrayBytes = $parameterBytes + $activationBytes
    $binaryBytes = (Get-Item $executable).Length
    $results.Add([pscustomobject]@{
        Precision = $variant.Name
        MedianNs = Get-Median $timings.ToArray()
        MaximumCodegenMismatch = $maximumCodegenMismatch
        MaximumF64Drift = $maximumF64Drift
        ParameterBytes = $parameterBytes
        FixedArrayBytes = $fixedArrayBytes
        BinaryBytes = $binaryBytes
    })
}

Write-Output "Compiler: $(& $compilerPath --version | Select-Object -First 1)"
Write-Output "Model: causal self-attention, tokens=$Tokens, features=$Width"
Write-Output "Timing: in-process generated loop; median of $Repetitions runs, $Iterations iterations each"
Write-Output "Fixed-array bytes include projection parameters, graph tensors, original input, and score scratch; they exclude executable/runtime overhead."
Write-Output "Precision  Median ns/inference  Generated/reference mismatch  Drift from f64  Parameters  Fixed arrays  Binary"
foreach ($result in $results) {
    Write-Output ("{0,-9} {1,19:F1} {2,29:G17} {3,15:G17} {4,11} {5,13} {6,8}" -f `
        $result.Precision, $result.MedianNs, $result.MaximumCodegenMismatch,
        $result.MaximumF64Drift, $result.ParameterBytes, $result.FixedArrayBytes,
        $result.BinaryBytes)
}
