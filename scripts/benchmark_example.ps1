param(
    [string] $Compiler = "clang++",
    [int] $Width = 96,
    [int] $Iterations = 1000,
    [int] $Repetitions = 7
)

$ErrorActionPreference = "Stop"
if ($Width -lt 2 -or $Iterations -lt 1 -or $Repetitions -lt 3) {
    throw "Use Width >= 2, Iterations >= 1, and Repetitions >= 3"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build\benchmark"
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
$compilerCommand = Get-Command $Compiler -ErrorAction Stop
$compilerPath = $compilerCommand.Source
$compilerDirectory = Split-Path -Parent $compilerPath
$env:PATH = "$compilerDirectory;$env:PATH"
$proofline = Join-Path $buildDirectory "proofline.exe"
$model = Join-Path $buildDirectory "dense-benchmark.proof"
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
$compilerSources = @("main.cpp", "attention_cache.cpp", "model.cpp", "certificate.cpp", "codegen.cpp") |
    ForEach-Object { Join-Path $projectRoot "src\$_" }

$modelLines = [System.Collections.Generic.List[string]]::new()
$denseLayerLines = [System.Collections.Generic.List[string]]::new()
$modelLines.Add("input $Width f64")
for ($layer = 0; $layer -lt 3; $layer++) {
    $weights = [System.Collections.Generic.List[string]]::new()
    $biases = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt ($Width * $Width); $index++) {
        $raw = (($index * 17 + $layer * 13) % 101) - 50
        $weights.Add(([double]$raw / 512.0).ToString("R", [Globalization.CultureInfo]::InvariantCulture))
    }
    for ($index = 0; $index -lt $Width; $index++) {
        $raw = (($index * 7 + $layer) % 17) - 8
        $biases.Add(([double]$raw / 256.0).ToString("R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $denseLine = "dense $Width f64 " + ($weights -join " ") + " " + ($biases -join " ")
    $denseLayerLines.Add($denseLine)
    $modelLines.Add($denseLine)
    if ($layer -lt 2) { $modelLines.Add("relu") }
}
[System.IO.File]::WriteAllLines($model, $modelLines, $utf8WithoutBom)

$sampleInputs = [System.Collections.Generic.List[string]]::new()
for ($index = 0; $index -lt $Width; $index++) {
    $raw = (($index * 5) % 19) - 9
    $sampleInputs.Add(([double]$raw / 16.0).ToString("R", [Globalization.CultureInfo]::InvariantCulture))
}

function ConvertTo-NumericValues([string] $Text) {
    $tokens = $Text.Trim() -split '[,\s]+' | Where-Object { $_ }
    return @($tokens | ForEach-Object {
        [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture)
    })
}

& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 @compilerSources -o $proofline
if ($LASTEXITCODE -ne 0) { throw "Proofline build failed" }

$variants = @(
    @{ Name = "f64"; Flag = @() },
    @{ Name = "bf16"; Flag = @("--bf16") }
)
$benchmarkPrograms = [System.Collections.Generic.List[object]]::new()
$layerBenchmarkPrograms = [System.Collections.Generic.List[object]]::new()
foreach ($variant in $variants) {
    $source = Join-Path $buildDirectory ($variant.Name + ".cpp")
    $executable = Join-Path $buildDirectory ($variant.Name + ".exe")
    $arguments = @("compile", $model, $source) + $variant.Flag
    & $proofline @arguments
    if ($LASTEXITCODE -ne 0) { throw "Code generation failed for $($variant.Name)" }
    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $source -o $executable
    if ($LASTEXITCODE -ne 0) { throw "Generated program build failed for $($variant.Name)" }
    $referenceArguments = @("run", $model)
    if ($variant.Name -eq "bf16") { $referenceArguments += "--bf16" }
    $referenceArguments += $sampleInputs.ToArray()
    $referenceOutput = & $proofline @referenceArguments
    if ($LASTEXITCODE -ne 0) { throw "Reference run failed for $($variant.Name)" }
    $referenceValues = ConvertTo-NumericValues (($referenceOutput -split ": ", 2)[-1])
    $generatedOutput = & $executable @sampleInputs
    if ($LASTEXITCODE -ne 0) { throw "Generated output check failed for $($variant.Name)" }
    $generatedValues = ConvertTo-NumericValues $generatedOutput
    if ($referenceValues.Count -ne $generatedValues.Count) {
        throw "Output width mismatch for $($variant.Name)"
    }
    for ($valueIndex = 0; $valueIndex -lt $referenceValues.Count; $valueIndex++) {
        if ($referenceValues[$valueIndex] -ne $generatedValues[$valueIndex]) {
            throw "Generated $($variant.Name) output differs at element $valueIndex"
        }
    }
    $benchmarkPrograms.Add([pscustomobject]@{
        Name = $variant.Name
        Kind = "whole-network"
        Layer = -1
        Executable = $executable
        Timings = [System.Collections.Generic.List[double]]::new()
    })
}

for ($layer = 0; $layer -lt $denseLayerLines.Count; $layer++) {
    $layerModel = Join-Path $buildDirectory "dense-layer-$layer.proof"
    [System.IO.File]::WriteAllLines($layerModel,
        @("input $Width f64", $denseLayerLines[$layer]), $utf8WithoutBom)
    foreach ($variant in $variants) {
        $programName = "dense_${layer}_$($variant.Name)"
        $source = Join-Path $buildDirectory "$programName.cpp"
        $executable = Join-Path $buildDirectory "$programName.exe"
        $arguments = @("compile", $layerModel, $source) + $variant.Flag
        & $proofline @arguments
        if ($LASTEXITCODE -ne 0) { throw "Code generation failed for $programName" }
        & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $source -o $executable
        if ($LASTEXITCODE -ne 0) { throw "Generated program build failed for $programName" }

        $referenceArguments = @("run", $layerModel)
        if ($variant.Name -eq "bf16") { $referenceArguments += "--bf16" }
        $referenceArguments += $sampleInputs.ToArray()
        $reference = & $proofline @referenceArguments
        if ($LASTEXITCODE -ne 0) { throw "Reference run failed for $programName" }
        $expectedValues = ConvertTo-NumericValues (($reference -split ": ", 2)[-1])
        $generated = & $executable @sampleInputs
        if ($LASTEXITCODE -ne 0) { throw "Generated output check failed for $programName" }
        $actualValues = ConvertTo-NumericValues $generated
        if ($expectedValues.Count -ne $actualValues.Count) {
            throw "Output width mismatch for $programName"
        }
        for ($valueIndex = 0; $valueIndex -lt $expectedValues.Count; $valueIndex++) {
            if ($expectedValues[$valueIndex] -ne $actualValues[$valueIndex]) {
                throw "Generated output differs at element $valueIndex for $programName"
            }
        }
        $layerBenchmarkPrograms.Add([pscustomobject]@{
            Name = $variant.Name
            Kind = "single-dense-layer"
            Layer = $layer
            Executable = $executable
            Timings = [System.Collections.Generic.List[double]]::new()
        })
    }
}

$allBenchmarkPrograms = @($benchmarkPrograms.ToArray()) + @($layerBenchmarkPrograms.ToArray())

foreach ($program in $allBenchmarkPrograms) {
    $null = & $program.Executable --bench $Iterations @sampleInputs
    if ($LASTEXITCODE -ne 0) { throw "Warm-up failed for $($program.Name)" }
}
for ($repetition = 0; $repetition -lt $Repetitions; $repetition++) {
    $runOrder = @($allBenchmarkPrograms)
    if (($repetition % 2) -eq 1) { [array]::Reverse($runOrder) }
    foreach ($program in $runOrder) {
        $measurement = & $program.Executable --bench $Iterations @sampleInputs
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $($program.Name)" }
        if ($measurement -notmatch "ns_per_inference=([0-9.eE+-]+)") {
            throw "Could not read timing from $($program.Name): $measurement"
        }
        $program.Timings.Add([double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture))
    }
}

$results = [System.Collections.Generic.List[object]]::new()
$layerResults = [System.Collections.Generic.List[object]]::new()
foreach ($program in $benchmarkPrograms) {
    $ordered = @($program.Timings | Sort-Object)
    $median = $ordered[[int][math]::Floor($ordered.Count / 2)]
    $results.Add([pscustomobject]@{
        Precision = $program.Name
        MedianNanoseconds = $median
        Repetitions = $Repetitions
        IterationsPerRepetition = $Iterations
        BinaryBytes = (Get-Item $program.Executable).Length
    })
}
foreach ($program in $layerBenchmarkPrograms) {
    $ordered = @($program.Timings | Sort-Object)
    $median = $ordered[[int][math]::Floor($ordered.Count / 2)]
    $layerResults.Add([pscustomobject]@{
        DenseLayer = $program.Layer
        Precision = $program.Name
        MedianNanoseconds = $median
        Repetitions = $Repetitions
        IterationsPerRepetition = $Iterations
        BinaryBytes = (Get-Item $program.Executable).Length
    })
}

Write-Output "Compiler: $((& $compilerPath --version | Select-Object -First 1))"
Write-Output "Processor identifier: $env:PROCESSOR_IDENTIFIER"
Write-Output "Logical processors reported: $([System.Environment]::ProcessorCount)"
Write-Output "Generated Dense model: width=$Width layers=3; deterministic weights and input"
Write-Output "Timing: steady-state loop inside generated executable; median of $Repetitions runs"
$results | Format-Table -AutoSize
Write-Output "Isolated Dense-layer measurements (each layer compiled as a one-layer graph):"
$layerResults | Format-Table -AutoSize
$comparisonLines = & $proofline compare $model @sampleInputs
$maximumError = 0.0
foreach ($line in $comparisonLines) {
    if ($line -match "absolute_error=([0-9.eE+-]+)") {
        $observedError = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
        if ($observedError -gt $maximumError) { $maximumError = $observedError }
    }
}
$payloadLine = $comparisonLines | Where-Object { $_ -like "tensor_payload_bytes:*" }
Write-Output "Maximum absolute output error on benchmark input: $maximumError"
Write-Output $payloadLine
