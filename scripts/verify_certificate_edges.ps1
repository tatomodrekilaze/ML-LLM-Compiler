param(
    [string] $Compiler = "clang++"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build\certificate-edges"
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

$cases = @(
    [pscustomobject]@{
        Name = "tie-boundary"
        Model = "tie-boundary"
        Domain = "tie-boundary"
        Plan = "tie-boundary"
        Samples = @("1")
    },
    [pscustomobject]@{
        Name = "cancellation"
        Model = "cancellation"
        Domain = "cancellation"
        Plan = "cancellation"
        Samples = @("1 1 1", "1 -1 1", "-1 1 -1", "0.5 0.5 0.5")
    },
    [pscustomobject]@{
        Name = "relu-crossing"
        Model = "relu-crossing"
        Domain = "relu-crossing"
        Plan = "relu-crossing"
        Samples = @("-1", "-0.125", "0", "0.125", "1")
    },
    [pscustomobject]@{
        Name = "tiny-mixed"
        Model = "tiny"
        Domain = "tiny"
        Plan = "tiny-mixed"
        Samples = @("0.1 -0.1 0.1", "-0.1 0 0.1", "0.05 -0.075 0.025")
    },
    [pscustomobject]@{
        Name = "tiny-f64"
        Model = "tiny"
        Domain = "tiny"
        Plan = "tiny-f64"
        Samples = @("0.1 -0.1 0.1", "-0.1 0 0.1", "0.05 -0.075 0.025")
    }
)

foreach ($case in $cases) {
    $model = Join-Path $projectRoot "examples\$($case.Model).proof"
    $domain = Join-Path $projectRoot "examples\$($case.Domain).domain"
    $plan = Join-Path $projectRoot "examples\$($case.Plan).plan"
    $report = Join-Path $buildDirectory "$($case.Name)-certificate.txt"
    & $proofline certify $model $plan $domain 1e100 $report
    if ($LASTEXITCODE -ne 0) { throw "Certificate failed for $($case.Name)" }
    $reportText = Get-Content $report -Raw
    $match = [regex]::Match($reportText,
        "Maximum certified f64-reference difference: ([0-9.eE+-]+)")
    if (-not $match.Success) { throw "No error bound found for $($case.Name)" }
    $bound = [double]::Parse($match.Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture)

    $generatedSource = Join-Path $buildDirectory "$($case.Name).cpp"
    $generatedProgram = Join-Path $buildDirectory "$($case.Name).exe"
    & $proofline compile $model $generatedSource --plan $plan
    if ($LASTEXITCODE -ne 0) { throw "Code generation failed for $($case.Name)" }
    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 `
        $generatedSource -o $generatedProgram
    if ($LASTEXITCODE -ne 0) { throw "Generated code build failed for $($case.Name)" }

    $maximumObserved = 0.0
    foreach ($sampleText in $case.Samples) {
        $sample = $sampleText.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
        $referenceLine = & $proofline run $model @sample
        if ($LASTEXITCODE -ne 0) { throw "Reference failed for $($case.Name): $sampleText" }
        $generatedLine = & $generatedProgram @sample
        if ($LASTEXITCODE -ne 0) { throw "Generated program failed for $($case.Name): $sampleText" }
        $referenceValue = [double](($referenceLine -split ": ")[-1])
        $generatedValue = [double]$generatedLine
        $observedError = [Math]::Abs($referenceValue - $generatedValue)
        $maximumObserved = [Math]::Max($maximumObserved, $observedError)
        if ($observedError -gt $bound) {
            throw "Observed output error exceeded certificate for $($case.Name): $sampleText"
        }
    }

    $grid = & $proofline check-grid $model $plan $domain 7
    if ($LASTEXITCODE -ne 0) { throw "Grid sanity check failed for $($case.Name)" }
    Write-Output "$($case.Name): bound=$bound, maximum targeted generated-code error=$maximumObserved"
    Write-Output $grid
}

$overflowModel = Join-Path $projectRoot "examples\bf16-overflow.proof"
$overflowDomain = Join-Path $projectRoot "examples\bf16-overflow.domain"
$overflowPlan = Join-Path $projectRoot "examples\bf16-overflow.plan"
$overflowReport = Join-Path $buildDirectory "bf16-overflow-certificate.txt"
$overflowOutput = & $proofline certify $overflowModel $overflowPlan `
    $overflowDomain 1e100 $overflowReport 2>&1
if ($LASTEXITCODE -eq 0) {
    throw "Certificate accepted a BF16 conversion that can overflow"
}
Write-Output "BF16 overflow case rejected as expected"
