param(
    [string] $Compiler = "clang++"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build\verification"
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

$compilerCommand = Get-Command $Compiler -ErrorAction Stop
$compilerPath = $compilerCommand.Source
$compilerDirectory = Split-Path -Parent $compilerPath
$env:PATH = "$compilerDirectory;$env:PATH"
$proofline = Join-Path $buildDirectory "proofline.exe"
$model = Join-Path $projectRoot "examples\tiny.proof"
$compilerSources = @("main.cpp", "attention_cache.cpp", "model.cpp", "certificate.cpp", "codegen.cpp") |
    ForEach-Object { Join-Path $projectRoot "src\$_" }

& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 @compilerSources -o $proofline
if ($LASTEXITCODE -ne 0) { throw "Proofline build failed" }

$cacheCheck = Join-Path $buildDirectory "attention_cache_check.exe"
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 `
    -I (Join-Path $projectRoot "src") `
    (Join-Path $projectRoot "tests\attention_cache.cpp") `
    (Join-Path $projectRoot "src\attention_cache.cpp") -o $cacheCheck
if ($LASTEXITCODE -ne 0) { throw "Attention cache check build failed" }
$cacheCheckOutput = & $cacheCheck
if ($LASTEXITCODE -ne 0) { throw "Attention cache check failed" }
Write-Output $cacheCheckOutput

$referenceText = & $proofline run $model 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Reference evaluation failed" }
$comparisonText = & $proofline compare $model 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Precision comparison failed" }

$generatedPrograms = @(
    @{ Name = "tiny_f64"; Flag = @() },
    @{ Name = "tiny_bf16"; Flag = @("--bf16") }
)

foreach ($program in $generatedPrograms) {
    $sourcePath = Join-Path $buildDirectory ($program.Name + ".cpp")
    $executablePath = Join-Path $buildDirectory ($program.Name + ".exe")
    $compileArguments = @("compile", $model, $sourcePath) + $program.Flag
    & $proofline @compileArguments
    if ($LASTEXITCODE -ne 0) { throw "Code generation failed for $($program.Name)" }

    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $sourcePath -o $executablePath
    if ($LASTEXITCODE -ne 0) { throw "Generated program build failed for $($program.Name)" }

    $generatedOutput = & $executablePath 1 2 3
    if ($LASTEXITCODE -ne 0) { throw "Generated program failed for $($program.Name)" }
    Write-Output "$($program.Name): $generatedOutput"
}

$planPath = Join-Path $buildDirectory "tiny.plan"
& $proofline plan $model (Join-Path $projectRoot "examples\tiny.cal") 0.01 $planPath
if ($LASTEXITCODE -ne 0) { throw "Precision planning failed" }

$plannedReference = & $proofline run $model --plan $planPath 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Planned reference evaluation failed" }
$plannedSource = Join-Path $buildDirectory "tiny_planned.cpp"
$plannedExecutable = Join-Path $buildDirectory "tiny_planned.exe"
& $proofline compile $model $plannedSource --plan $planPath
if ($LASTEXITCODE -ne 0) { throw "Planned code generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $plannedSource -o $plannedExecutable
if ($LASTEXITCODE -ne 0) { throw "Planned program build failed" }
$plannedOutput = & $plannedExecutable 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Planned program failed" }
$plannedReferenceValue = ($plannedReference -split ": ")[-1]
if ($plannedOutput -ne $plannedReferenceValue) {
    throw "Generated planned output differs from the planned reference"
}
foreach ($sampleLine in Get-Content (Join-Path $projectRoot "examples\tiny.cal")) {
    $sampleLine = ($sampleLine -split "#", 2)[0].Trim()
    if (-not $sampleLine) { continue }
    $sample = $sampleLine.Split([char[]]@(' ', "`t"), [StringSplitOptions]::RemoveEmptyEntries)
    $sampleReference = & $proofline run $model --plan $planPath @sample
    if ($LASTEXITCODE -ne 0) { throw "Planned evaluator failed on calibration input" }
    $sampleOutput = & $plannedExecutable @sample
    if ($LASTEXITCODE -ne 0) { throw "Generated plan failed on calibration input" }
    $sampleReferenceValue = ($sampleReference -split ": ")[-1]
    if ($sampleOutput -ne $sampleReferenceValue) {
        throw "Generated plan differs from the reference on calibration input: $sampleLine"
    }
}
$bf16Executable = Join-Path $buildDirectory "tiny_bf16.exe"
$random = [System.Random]::new(137)
for ($case = 0; $case -lt 64; $case++) {
    $sample = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt 3; $index++) {
        $value = ($random.NextDouble() * 10.0) - 5.0
        $sample.Add($value.ToString("R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $referenceLine = & $proofline run $model --bf16 @sample
    if ($LASTEXITCODE -ne 0) { throw "Bfloat16 reference failed on deterministic random input" }
    $generatedLine = & $bf16Executable @sample
    if ($LASTEXITCODE -ne 0) { throw "Generated bfloat16 program failed on deterministic random input" }
    $referenceValue = [double](($referenceLine -split ": ")[-1])
    $generatedValue = [double]$generatedLine
    if ($referenceValue -ne $generatedValue) {
        throw "Generated bfloat16 output differs from the reference on random case $case"
    }
}
Write-Output "bfloat16 reference/codegen matched on 64 deterministic random inputs"
Write-Output "planned output: $plannedOutput"

$rmsNormModel = Join-Path $projectRoot "examples\rmsnorm.proof"
$rmsNormDescription = & $proofline inspect $rmsNormModel
if ($LASTEXITCODE -ne 0 -or ($rmsNormDescription -join "`n") -notmatch "RMSNorm") {
    throw "RMSNorm model parsing or graph inspection failed"
}
$rmsNormReferenceLine = & $proofline run $rmsNormModel 3 4 0
if ($LASTEXITCODE -ne 0) { throw "RMSNorm reference evaluation failed" }
$rmsNormReference = (($rmsNormReferenceLine -split ": ")[-1] -split '[,\s]+' |
    Where-Object { $_ }) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$rmsNormSource = Join-Path $buildDirectory "rmsnorm.cpp"
$rmsNormExecutable = Join-Path $buildDirectory "rmsnorm.exe"
& $proofline compile $rmsNormModel $rmsNormSource
if ($LASTEXITCODE -ne 0) { throw "RMSNorm C++ generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $rmsNormSource -o $rmsNormExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated RMSNorm program build failed" }
$rmsNormOutputLine = & $rmsNormExecutable 3 4 0
if ($LASTEXITCODE -ne 0) { throw "Generated RMSNorm program failed" }
$rmsNormOutput = $rmsNormOutputLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
if ($rmsNormOutput.Count -ne 3 -or $rmsNormReference.Count -ne 3) {
    throw "RMSNorm output width was not preserved"
}
for ($index = 0; $index -lt 3; $index++) {
    if ([Math]::Abs($rmsNormOutput[$index] - $rmsNormReference[$index]) -gt 1.0e-14) {
        throw "Generated RMSNorm output differs from the reference at element $index"
    }
}
$rmsNormDenominator = [Math]::Sqrt((25.0 / 3.0) + 0.00001)
$expectedRmsNorm = [double[]]@(
    (3.0 / $rmsNormDenominator),
    (4.0 / $rmsNormDenominator * 2.0),
    0.0
)
for ($index = 0; $index -lt 3; $index++) {
    if ([Math]::Abs($rmsNormOutput[$index] - $expectedRmsNorm[$index]) -gt 1.0e-14) {
        throw "RMSNorm output differs from the hand-calculated formula at element $index"
    }
}
$rmsNormRandom = [System.Random]::new(9321)
for ($case = 0; $case -lt 16; $case++) {
    $sample = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt 3; $index++) {
        $value = ($rmsNormRandom.NextDouble() * 8.0) - 4.0
        $sample.Add($value.ToString("R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $referenceLine = & $proofline run $rmsNormModel @sample
    if ($LASTEXITCODE -ne 0) { throw "RMSNorm reference failed on deterministic sample $case" }
    $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
        Where-Object { $_ }) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $generatedLine = & $rmsNormExecutable @sample
    if ($LASTEXITCODE -ne 0) { throw "Generated RMSNorm program failed on sample $case" }
    $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    for ($index = 0; $index -lt 3; $index++) {
        if ([Math]::Abs($referenceValues[$index] - $generatedValues[$index]) -gt 1.0e-14) {
            throw "Generated RMSNorm output differs from the reference on sample $case"
        }
    }
}
$largeInput = @("1e308", "1e308", "1e308")
$largeReferenceLine = & $proofline run $rmsNormModel @largeInput
if ($LASTEXITCODE -ne 0) { throw "RMSNorm reference failed on large finite inputs" }
$largeReference = (($largeReferenceLine -split ": ")[-1] -split '[,\s]+' |
    Where-Object { $_ }) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$largeGeneratedLine = & $rmsNormExecutable @largeInput
if ($LASTEXITCODE -ne 0) { throw "Generated RMSNorm program failed on large finite inputs" }
$largeGenerated = $largeGeneratedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$largeExpected = @(1.0, 2.0, 0.5)
for ($index = 0; $index -lt 3; $index++) {
    $referenceError = [Math]::Abs($largeReference[$index] - $largeExpected[$index])
    $generatedError = [Math]::Abs($largeGenerated[$index] - $largeExpected[$index])
    if ($referenceError -gt 1.0e-14 -or $generatedError -gt 1.0e-14) {
        throw "Scaled RMSNorm failed on large finite input at element $index"
    }
}
Write-Output "RMSNorm matched the formula on one hand calculation, 16 deterministic samples, and 1e308 inputs"
$rmsNormPlan = Join-Path $buildDirectory "rmsnorm.plan"
$rmsNormDomain = Join-Path $buildDirectory "rmsnorm.domain"
$rmsNormCertificate = Join-Path $buildDirectory "rmsnorm-certificate.txt"
Set-Content -Path $rmsNormPlan -Value "# RMSNorm has no Dense precision decisions" -Encoding ascii
Set-Content -Path $rmsNormDomain -Value @("-1 1", "-1 1", "-1 1") -Encoding ascii
& $proofline certify $rmsNormModel $rmsNormPlan $rmsNormDomain 1.0 $rmsNormCertificate
if ($LASTEXITCODE -ne 1) {
    throw "Bounded-domain certification did not reject the unsupported RMSNorm operation"
}
Write-Output "Certificate correctly rejects RMSNorm until its bound is implemented"

$residualModel = Join-Path $projectRoot "examples\residual.proof"
$residualDescription = & $proofline inspect $residualModel
if ($LASTEXITCODE -ne 0 -or ($residualDescription -join "`n") -notmatch "v1 \+ input") {
    throw "Residual model parsing or multi-input graph inspection failed"
}
$residualSource = Join-Path $buildDirectory "residual.cpp"
$residualExecutable = Join-Path $buildDirectory "residual.exe"
& $proofline compile $residualModel $residualSource --bf16
if ($LASTEXITCODE -ne 0) { throw "Residual C++ generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $residualSource -o $residualExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated residual program build failed" }
$residualReferenceLine = & $proofline run $residualModel --bf16 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Residual reference evaluation failed" }
$residualReference = (($residualReferenceLine -split ": ")[-1] -split '[,\s]+' |
    Where-Object { $_ }) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$residualGeneratedLine = & $residualExecutable 1 2 3
if ($LASTEXITCODE -ne 0) { throw "Generated residual program failed" }
$residualGenerated = $residualGeneratedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$residualExpected = @(2.0, 4.0, 6.0)
for ($index = 0; $index -lt 3; $index++) {
    $referenceError = [Math]::Abs($residualReference[$index] - $residualExpected[$index])
    $generatedError = [Math]::Abs($residualGenerated[$index] - $residualExpected[$index])
    if ($referenceError -gt 1.0e-14 -or $generatedError -gt 1.0e-14) {
        throw "Residual Add output is incorrect at element $index"
    }
}
$residualPlan = Join-Path $buildDirectory "residual.plan"
$residualCertificate = Join-Path $buildDirectory "residual-certificate.txt"
Set-Content -Path $residualPlan -Value "dense bf16" -Encoding ascii
$residualDomain = Join-Path $projectRoot "examples\residual.domain"
& $proofline certify $residualModel $residualPlan $residualDomain 1.0 $residualCertificate
if ($LASTEXITCODE -ne 0) { throw "Residual graph certificate failed" }
$residualGrid = & $proofline check-grid $residualModel $residualPlan $residualDomain 3
if ($LASTEXITCODE -ne 0) { throw "Residual certificate grid check failed" }
Write-Output "Multi-input residual graph matched 2x input and passed a 27-point certificate grid"

$branchesModel = Join-Path $projectRoot "examples\branches.proof"
$branchesDescription = & $proofline inspect $branchesModel
if ($LASTEXITCODE -ne 0 -or ($branchesDescription -join "`n") -notmatch "Operation 1: Dense input -> v2") {
    throw "Branched graph parser or graph inspection failed"
}
$branchesSource = Join-Path $buildDirectory "branches.cpp"
$branchesExecutable = Join-Path $buildDirectory "branches.exe"
& $proofline compile $branchesModel $branchesSource --bf16
if ($LASTEXITCODE -ne 0) { throw "Branched graph C++ generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $branchesSource -o $branchesExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated branched graph program build failed" }
$branchesOutputLine = & $branchesExecutable 1 2
if ($LASTEXITCODE -ne 0) { throw "Generated branched graph program failed" }
if ($branchesOutputLine -ne "3 8") {
    throw "The two Dense branches did not combine to the expected output: $branchesOutputLine"
}
$branchesPlan = Join-Path $buildDirectory "branches.plan"
$branchesCertificate = Join-Path $buildDirectory "branches-certificate.txt"
Set-Content -Path $branchesPlan -Value @("dense bf16", "dense bf16") -Encoding ascii
$branchesDomain = Join-Path $projectRoot "examples\branches.domain"
& $proofline certify $branchesModel $branchesPlan $branchesDomain 1.0 $branchesCertificate
if ($LASTEXITCODE -ne 0) { throw "Branched graph certificate failed" }
$branchesGrid = & $proofline check-grid $branchesModel $branchesPlan $branchesDomain 3
if ($LASTEXITCODE -ne 0) { throw "Branched graph certificate grid check failed" }
Write-Output "Two Dense branches from one input generated 3 8 and passed a 9-point certificate grid"

$softmaxModel = Join-Path $projectRoot "examples\softmax.proof"
$softmaxDescription = & $proofline inspect $softmaxModel
if ($LASTEXITCODE -ne 0 -or ($softmaxDescription -join "`n") -notmatch "Softmax") {
    throw "Softmax model parsing or graph inspection failed"
}
$softmaxSource = Join-Path $buildDirectory "softmax.cpp"
$softmaxExecutable = Join-Path $buildDirectory "softmax.exe"
& $proofline compile $softmaxModel $softmaxSource
if ($LASTEXITCODE -ne 0) { throw "Softmax C++ generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $softmaxSource -o $softmaxExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated Softmax program build failed" }
$softmaxExpectedScores = @(
    @{ Input = @("1000", "999", "-1000"); Expected = @((1.0 / (1.0 + [Math]::Exp(-1.0))), ([Math]::Exp(-1.0) / (1.0 + [Math]::Exp(-1.0))), 0.0) },
    @{ Input = @("1e308", "1e308", "-1e308"); Expected = @(0.5, 0.5, 0.0) }
)
foreach ($case in $softmaxExpectedScores) {
    $softmaxInput = $case.Input
    $referenceLine = & $proofline run $softmaxModel @softmaxInput
    if ($LASTEXITCODE -ne 0) { throw "Softmax reference failed on stable-softmax fixture" }
    $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
        Where-Object { $_ }) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $generatedLine = & $softmaxExecutable @softmaxInput
    if ($LASTEXITCODE -ne 0) { throw "Generated Softmax program failed on stable-softmax fixture" }
    $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $probabilitySum = 0.0
    for ($index = 0; $index -lt 3; $index++) {
        $referenceError = [Math]::Abs($referenceValues[$index] - $case.Expected[$index])
        $generatedError = [Math]::Abs($generatedValues[$index] - $case.Expected[$index])
        if ($referenceError -gt 1.0e-14 -or $generatedError -gt 1.0e-14) {
            throw "Softmax output differs from expected probability at element $index"
        }
        $probabilitySum += $generatedValues[$index]
    }
    if ([Math]::Abs($probabilitySum - 1.0) -gt 1.0e-14) {
        throw "Softmax probabilities do not sum to one"
    }
}
$softmaxRandom = [System.Random]::new(4107)
for ($case = 0; $case -lt 16; $case++) {
    $sample = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt 3; $index++) {
        $sample.Add((($softmaxRandom.NextDouble() * 2000.0) - 1000.0).ToString(
            "R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $referenceLine = & $proofline run $softmaxModel @sample
    if ($LASTEXITCODE -ne 0) { throw "Softmax reference failed on deterministic sample $case" }
    $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
        Where-Object { $_ }) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $generatedLine = & $softmaxExecutable @sample
    if ($LASTEXITCODE -ne 0) { throw "Generated Softmax program failed on sample $case" }
    $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    for ($index = 0; $index -lt 3; $index++) {
        if ([Math]::Abs($referenceValues[$index] - $generatedValues[$index]) -gt 1.0e-14) {
            throw "Generated Softmax output differs from reference on sample $case"
        }
    }
}
Write-Output "Softmax matched expected probabilities, 1e308 logits, and 16 deterministic samples"
$softmaxPlan = Join-Path $buildDirectory "softmax.plan"
$softmaxDomain = Join-Path $buildDirectory "softmax.domain"
$softmaxCertificate = Join-Path $buildDirectory "softmax-certificate.txt"
Set-Content -Path $softmaxPlan -Value "# Softmax has no Dense precision decisions" -Encoding ascii
Set-Content -Path $softmaxDomain -Value @("-1 1", "-1 1", "-1 1") -Encoding ascii
& $proofline certify $softmaxModel $softmaxPlan $softmaxDomain 1.0 $softmaxCertificate
if ($LASTEXITCODE -ne 1) {
    throw "Bounded-domain certification did not reject the unsupported Softmax operation"
}
Write-Output "Certificate correctly rejects Softmax until its bound is implemented"

$ropeModel = Join-Path $projectRoot "examples\rope.proof"
$ropeDescription = & $proofline inspect $ropeModel
if ($LASTEXITCODE -ne 0 -or ($ropeDescription -join "`n") -notmatch "RoPE") {
    throw "RoPE model parsing or graph inspection failed"
}
$ropeSource = Join-Path $buildDirectory "rope.cpp"
$ropeExecutable = Join-Path $buildDirectory "rope.exe"
& $proofline compile $ropeModel $ropeSource
if ($LASTEXITCODE -ne 0) { throw "RoPE C++ generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $ropeSource -o $ropeExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated RoPE program build failed" }
$ropeInput = @("1", "0", "1", "0")
$ropeReferenceLine = & $proofline run $ropeModel @ropeInput
if ($LASTEXITCODE -ne 0) { throw "RoPE reference evaluation failed" }
$ropeReference = (($ropeReferenceLine -split ": ")[-1] -split '[,\s]+' |
    Where-Object { $_ }) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$ropeGeneratedLine = & $ropeExecutable @ropeInput
if ($LASTEXITCODE -ne 0) { throw "Generated RoPE program failed" }
$ropeGenerated = $ropeGeneratedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$ropeExpected = @([Math]::Cos(1.0), [Math]::Sin(1.0), [Math]::Cos(0.01), [Math]::Sin(0.01))
for ($index = 0; $index -lt 4; $index++) {
    $referenceError = [Math]::Abs($ropeReference[$index] - $ropeExpected[$index])
    $generatedError = [Math]::Abs($ropeGenerated[$index] - $ropeExpected[$index])
    if ($referenceError -gt 1.0e-14 -or $generatedError -gt 1.0e-14) {
        throw "RoPE output differs from the hand-calculated rotation at element $index"
    }
}
$ropeRandom = [System.Random]::new(7813)
for ($case = 0; $case -lt 16; $case++) {
    $sample = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt 4; $index++) {
        $value = ($ropeRandom.NextDouble() * 8.0) - 4.0
        $sample.Add($value.ToString("R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $referenceLine = & $proofline run $ropeModel @sample
    if ($LASTEXITCODE -ne 0) { throw "RoPE reference failed on deterministic sample $case" }
    $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
        Where-Object { $_ }) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $generatedLine = & $ropeExecutable @sample
    if ($LASTEXITCODE -ne 0) { throw "Generated RoPE program failed on deterministic sample $case" }
    $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    for ($index = 0; $index -lt 4; $index++) {
        if ([Math]::Abs($referenceValues[$index] - $generatedValues[$index]) -gt 1.0e-14) {
            throw "Generated RoPE output differs from reference on sample $case"
        }
    }
}
$largeRopeModel = Join-Path $buildDirectory "rope-large-position.proof"
Set-Content -Path $largeRopeModel -Value @("input 4 f64", "rope 1000000000 10000") -Encoding ascii
$largeRopeSource = Join-Path $buildDirectory "rope-large-position.cpp"
$largeRopeExecutable = Join-Path $buildDirectory "rope-large-position.exe"
& $proofline compile $largeRopeModel $largeRopeSource
if ($LASTEXITCODE -ne 0) { throw "Large-position RoPE code generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $largeRopeSource -o $largeRopeExecutable
if ($LASTEXITCODE -ne 0) { throw "Large-position RoPE program build failed" }
$largeRopeInput = @("1", "0", "1", "0")
$largeRopeReference = & $proofline run $largeRopeModel @largeRopeInput
if ($LASTEXITCODE -ne 0) { throw "Large-position RoPE reference failed" }
$largeRopeGenerated = & $largeRopeExecutable @largeRopeInput
if ($LASTEXITCODE -ne 0) { throw "Large-position RoPE generated program failed" }
$largeRopeReferenceValues = (($largeRopeReference -split ": ")[-1] -split '[,\s]+' |
    Where-Object { $_ }) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$largeRopeGeneratedValues = $largeRopeGenerated.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
for ($index = 0; $index -lt 4; $index++) {
    if ([Math]::Abs($largeRopeReferenceValues[$index] - $largeRopeGeneratedValues[$index]) -gt 1.0e-14) {
        throw "Large-position RoPE generated output differs from reference"
    }
}
$ropePlan = Join-Path $buildDirectory "rope.plan"
$ropeCertificate = Join-Path $buildDirectory "rope-certificate.txt"
$ropeDomain = Join-Path $projectRoot "examples\rope.domain"
Set-Content -Path $ropePlan -Value "# RoPE has no Dense precision decisions" -Encoding ascii
& $proofline certify $ropeModel $ropePlan $ropeDomain 1.0 $ropeCertificate
if ($LASTEXITCODE -ne 1) {
    throw "Bounded-domain certification did not reject unsupported RoPE"
}
Write-Output "RoPE matched two known rotations and 16 deterministic samples; certification refuses unsupported trig bounds"

$domainPath = Join-Path $projectRoot "examples\tiny.domain"
$certificatePath = Join-Path $buildDirectory "tiny-certificate.txt"
$certificate = & $proofline certify $model $planPath $domainPath 0.1 $certificatePath
if ($LASTEXITCODE -ne 0) { throw "Bounded-domain certification failed" }
Write-Output $certificate
$gridCheck = & $proofline check-grid $model $planPath $domainPath 11
if ($LASTEXITCODE -ne 0) { throw "Certificate grid sanity check failed" }
Write-Output $gridCheck

$certificateText = Get-Content $certificatePath -Raw
$certifiedBoundMatch = [regex]::Match($certificateText,
    "Maximum certified f64-reference difference: ([0-9.eE+-]+)")
if (-not $certifiedBoundMatch.Success) { throw "Certificate report has no maximum error bound" }
$certifiedBound = [double]::Parse($certifiedBoundMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture)
$maximumGeneratedGridError = 0.0
for ($a = 0; $a -lt 11; $a++) {
    for ($b = 0; $b -lt 11; $b++) {
        for ($c = 0; $c -lt 11; $c++) {
            $sample = @(
                ((-0.1 + 0.02 * $a).ToString("R", [Globalization.CultureInfo]::InvariantCulture)),
                ((-0.1 + 0.02 * $b).ToString("R", [Globalization.CultureInfo]::InvariantCulture)),
                ((-0.1 + 0.02 * $c).ToString("R", [Globalization.CultureInfo]::InvariantCulture))
            )
            if ($a -eq 0) { $sample[0] = "-0.1" }
            if ($a -eq 10) { $sample[0] = "0.1" }
            if ($b -eq 0) { $sample[1] = "-0.1" }
            if ($b -eq 10) { $sample[1] = "0.1" }
            if ($c -eq 0) { $sample[2] = "-0.1" }
            if ($c -eq 10) { $sample[2] = "0.1" }
            $sampleReference = & $proofline run $model @sample
            if ($LASTEXITCODE -ne 0) { throw "Reference evaluator failed on certificate grid" }
            $sampleGenerated = & $plannedExecutable @sample
            if ($LASTEXITCODE -ne 0) { throw "Generated program failed on certificate grid" }
            $referenceValue = [double](($sampleReference -split ": ")[-1])
            $generatedValue = [double]$sampleGenerated
            $observedError = [Math]::Abs($referenceValue - $generatedValue)
            $maximumGeneratedGridError = [Math]::Max($maximumGeneratedGridError, $observedError)
            if ($observedError -gt $certifiedBound) {
                throw "Generated output exceeded the certificate at grid point $a,$b,$c"
            }
        }
    }
}
Write-Output "Generated code checked on 1,331 domain points; maximum absolute error: $maximumGeneratedGridError"

$rejectedCertificatePath = Join-Path $buildDirectory "tiny-rejected-certificate.txt"
$rejectedCertificate = & $proofline certify $model $planPath $domainPath 0.001 $rejectedCertificatePath
if ($LASTEXITCODE -ne 1) { throw "Verifier did not reject a limit below its certified bound" }
if (-not (Select-String -Path $rejectedCertificatePath -Pattern "^Verdict: FAIL$" -Quiet)) {
    throw "Rejected certificate report does not contain a FAIL verdict"
}
Write-Output "Verifier correctly rejected an error limit below the certified bound"

Write-Output $referenceText
Write-Output $comparisonText

$sequenceModel = Join-Path $projectRoot "examples\two_token.proof"
$sequenceDescription = & $proofline inspect $sequenceModel
if ($LASTEXITCODE -ne 0 -or ($sequenceDescription -join "`n") -notmatch "tensor<2x2xf64>") {
    throw "Two-token matrix model did not preserve its tensor shape"
}
$sequenceExpected = "3 9 9 17"
$sequenceFlags = @(
    @{ Name = "sequence_f64"; Flag = @() },
    @{ Name = "sequence_bf16"; Flag = @("--bf16") }
)
foreach ($program in $sequenceFlags) {
    $sourcePath = Join-Path $buildDirectory ($program.Name + ".cpp")
    $executablePath = Join-Path $buildDirectory ($program.Name + ".exe")
    & $proofline compile $sequenceModel $sourcePath @($program.Flag)
    if ($LASTEXITCODE -ne 0) { throw "Two-token code generation failed for $($program.Name)" }
    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $sourcePath -o $executablePath
    if ($LASTEXITCODE -ne 0) { throw "Two-token generated build failed for $($program.Name)" }
    $actual = & $executablePath 1 2 3 4
    if ($LASTEXITCODE -ne 0 -or ($actual -join " ") -ne $sequenceExpected) {
        throw "Two-token generated output did not match the hand-calculated result"
    }
}
$sequenceReference = & $proofline run $sequenceModel 1 2 3 4
if ($LASTEXITCODE -ne 0 -or ($sequenceReference -join " ") -ne "Output (precision plan): 3, 9, 9, 17") {
    throw "Two-token reference output did not match the hand-calculated result"
}
$sequencePlan = Join-Path $buildDirectory "two-token.plan"
& $proofline plan $sequenceModel (Join-Path $projectRoot "examples\two_token.cal") 0.01 $sequencePlan
if ($LASTEXITCODE -ne 0) { throw "Two-token precision planning failed" }
$plannedSequenceSource = Join-Path $buildDirectory "two-token-planned.cpp"
$plannedSequenceExecutable = Join-Path $buildDirectory "two-token-planned.exe"
& $proofline compile $sequenceModel $plannedSequenceSource --plan $sequencePlan
if ($LASTEXITCODE -ne 0) { throw "Two-token planned code generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $plannedSequenceSource -o $plannedSequenceExecutable
if ($LASTEXITCODE -ne 0) { throw "Two-token planned generated build failed" }
$plannedSequence = & $plannedSequenceExecutable 1 2 3 4
if ($LASTEXITCODE -ne 0 -or ($plannedSequence -join " ") -ne $sequenceExpected) {
    throw "Two-token planned generated output did not match the hand-calculated result"
}
Write-Output "Two-token Linear, ReLU, and residual Add matched the hand calculation in f64 and BF16 paths"

$attentionScale = 1.0 / [Math]::Sqrt(2.0)
$laterTokenWeight = 1.0 / (1.0 + [Math]::Exp(-$attentionScale))
$earlierTokenWeight = 1.0 - $laterTokenWeight
$attentionCases = @(
    @{ Name = "causal"; Model = (Join-Path $projectRoot "examples\causal_attention.proof");
       Expected = @(1.0, 0.0, $earlierTokenWeight, $laterTokenWeight) },
    @{ Name = "full"; Model = (Join-Path $projectRoot "examples\full_attention.proof");
       Expected = @($laterTokenWeight, $earlierTokenWeight, $earlierTokenWeight, $laterTokenWeight) }
)
$attentionRandom = [System.Random]::new(18531)
foreach ($attentionCase in $attentionCases) {
    $description = & $proofline inspect $attentionCase.Model
    if ($LASTEXITCODE -ne 0 -or ($description -join "`n") -notmatch "Attention") {
        throw "The $($attentionCase.Name) attention graph did not parse"
    }
    foreach ($variant in @(
        @{ Name = "f64"; Flag = @() },
        @{ Name = "bf16"; Flag = @("--bf16") }
    )) {
        $stem = "attention-$($attentionCase.Name)-$($variant.Name)"
        $sourcePath = Join-Path $buildDirectory ($stem + ".cpp")
        $executablePath = Join-Path $buildDirectory ($stem + ".exe")
        & $proofline compile $attentionCase.Model $sourcePath @($variant.Flag)
        if ($LASTEXITCODE -ne 0) { throw "Attention code generation failed for $stem" }
        & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $sourcePath -o $executablePath
        if ($LASTEXITCODE -ne 0) { throw "Generated attention build failed for $stem" }

        $input = @("1", "0", "0", "1")
        $generatedLine = & $executablePath @input
        if ($LASTEXITCODE -ne 0) { throw "Generated attention execution failed for $stem" }
        $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
            ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
        for ($index = 0; $index -lt 4; $index++) {
            if ([Math]::Abs($generatedValues[$index] - $attentionCase.Expected[$index]) -gt 1.0e-14) {
                throw "$stem differs from the hand-calculated attention result at output $index"
            }
        }

        for ($sampleIndex = 0; $sampleIndex -lt 16; $sampleIndex++) {
            $sample = [System.Collections.Generic.List[string]]::new()
            for ($element = 0; $element -lt 4; $element++) {
                $sample.Add((($attentionRandom.NextDouble() * 2.0) - 1.0).ToString(
                    "R", [Globalization.CultureInfo]::InvariantCulture))
            }
            $referenceArguments = @("run", $attentionCase.Model)
            if ($variant.Name -eq "bf16") { $referenceArguments += "--bf16" }
            $referenceArguments += $sample.ToArray()
            $referenceLine = & $proofline @referenceArguments
            if ($LASTEXITCODE -ne 0) { throw "Attention reference failed on sample $sampleIndex" }
            $generatedLine = & $executablePath @sample
            if ($LASTEXITCODE -ne 0) { throw "Generated attention failed on sample $sampleIndex" }
            $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
                Where-Object { $_ }) |
                ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
            $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
                ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
            for ($index = 0; $index -lt 4; $index++) {
                if ([Math]::Abs($referenceValues[$index] - $generatedValues[$index]) -gt 1.0e-13) {
                    throw "$stem differs from the reference on sample $sampleIndex"
                }
            }
        }
    }
}
Write-Output "Causal and full attention matched hand calculations and 16 deterministic samples in f64 and BF16"

$multiHeadModel = Join-Path $projectRoot "examples\two_head_attention.proof"
$multiHeadDescription = & $proofline inspect $multiHeadModel
if ($LASTEXITCODE -ne 0 -or ($multiHeadDescription -join "`n") -notmatch "2 heads") {
    throw "The two-head attention model did not parse or report its head count"
}
$multiHeadExpected = @(1.0, 0.0, 0.0, 1.0, $earlierTokenWeight, $laterTokenWeight,
    $laterTokenWeight, $earlierTokenWeight)
foreach ($variant in @(
    @{ Name = "f64"; Flag = @() },
    @{ Name = "bf16"; Flag = @("--bf16") }
)) {
    $stem = "attention-two-head-$($variant.Name)"
    $sourcePath = Join-Path $buildDirectory ($stem + ".cpp")
    $executablePath = Join-Path $buildDirectory ($stem + ".exe")
    & $proofline compile $multiHeadModel $sourcePath @($variant.Flag)
    if ($LASTEXITCODE -ne 0) { throw "Multi-head code generation failed for $stem" }
    & $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $sourcePath -o $executablePath
    if ($LASTEXITCODE -ne 0) { throw "Generated multi-head build failed for $stem" }

    $input = @("1", "0", "0", "1", "0", "1", "1", "0")
    $generatedLine = & $executablePath @input
    if ($LASTEXITCODE -ne 0) { throw "Generated multi-head execution failed for $stem" }
    $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    for ($index = 0; $index -lt 8; $index++) {
        if ([Math]::Abs($generatedValues[$index] - $multiHeadExpected[$index]) -gt 1.0e-14) {
            throw "$stem differs from the hand-calculated result at output $index"
        }
    }

    for ($sampleIndex = 0; $sampleIndex -lt 16; $sampleIndex++) {
        $sample = [System.Collections.Generic.List[string]]::new()
        for ($element = 0; $element -lt 8; $element++) {
            $sample.Add((($attentionRandom.NextDouble() * 2.0) - 1.0).ToString(
                "R", [Globalization.CultureInfo]::InvariantCulture))
        }
        $referenceArguments = @("run", $multiHeadModel)
        if ($variant.Name -eq "bf16") { $referenceArguments += "--bf16" }
        $referenceArguments += $sample.ToArray()
        $referenceLine = & $proofline @referenceArguments
        if ($LASTEXITCODE -ne 0) { throw "Multi-head reference failed on sample $sampleIndex" }
        $generatedLine = & $executablePath @sample
        if ($LASTEXITCODE -ne 0) { throw "Generated multi-head run failed on sample $sampleIndex" }
        $referenceValues = (($referenceLine -split ": ")[-1] -split '[,\s]+' |
            Where-Object { $_ }) |
            ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
        $generatedValues = $generatedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
            ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
        for ($index = 0; $index -lt 8; $index++) {
            if ([Math]::Abs($referenceValues[$index] - $generatedValues[$index]) -gt 1.0e-13) {
                throw "$stem differs from the reference on sample $sampleIndex"
            }
        }
    }
}
Write-Output "Two-head causal attention matched a hand calculation and 16 deterministic samples in f64 and BF16"

$sessionHarness = Join-Path $buildDirectory "generated_attention_session.cpp"
@'
#define PROOFLINE_LIBRARY
#include "attention-two-head-f64.cpp"

#include <array>
#include <cmath>
#include <stdexcept>

int main() {
    AttentionSession_3 session;
    const AttentionSession_3::InputRow token0{1.0, 0.0, 0.0, 1.0};
    const auto first = session.stepFromInput(token0);
    const std::array<double, 4> firstExpected{1.0, 0.0, 0.0, 1.0};
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (std::abs(first[i] - firstExpected[i]) > 1.0e-14) return 1;
    }

    const AttentionSession_3::InputRow token1{0.0, 1.0, 1.0, 0.0};
    const auto second = session.stepFromInput(token1);
    const double laterWeight = 1.0 / (1.0 + std::exp(-1.0 / std::sqrt(2.0)));
    const std::array<double, 4> secondExpected{
        1.0 - laterWeight, laterWeight, laterWeight, 1.0 - laterWeight};
    for (std::size_t i = 0; i < second.size(); ++i) {
        if (std::abs(second[i] - secondExpected[i]) > 1.0e-14) return 2;
    }
    if (session.size() != 2 || AttentionSession_3::capacity() != 2) return 3;

    bool rejectedOverflow = false;
    try { (void)session.stepFromInput(token1); }
    catch (const std::length_error&) { rejectedOverflow = true; }
    if (!rejectedOverflow) return 4;

    session.reset();
    if (session.size() != 0) return 5;
    const auto afterReset = session.stepFromInput(token0);
    for (std::size_t i = 0; i < afterReset.size(); ++i) {
        if (std::abs(afterReset[i] - firstExpected[i]) > 1.0e-14) return 6;
    }

    AttentionSession_3 projectedSession;
    const AttentionSession_3::Query query0{1.0, 0.0, 0.0, 1.0};
    const AttentionSession_3::Key key0{1.0, 0.0, 0.0, 1.0};
    const AttentionSession_3::Value value0{1.0, 0.0, 0.0, 1.0};
    const auto projectedOutput = projectedSession.step(query0, key0, value0);
    for (std::size_t i = 0; i < projectedOutput.size(); ++i) {
        if (std::abs(projectedOutput[i] - firstExpected[i]) > 1.0e-14) return 7;
    }
    return 0;
}
'@ | Set-Content -Path $sessionHarness -Encoding ascii
$sessionExecutable = Join-Path $buildDirectory "generated_attention_session.exe"
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 $sessionHarness -o $sessionExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated persistent attention-session build failed" }
& $sessionExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated persistent attention-session checks failed" }
Write-Output "Generated C++ session preserved two-head KV state across calls and reset cleanly"
$bf16SessionHarness = Join-Path $buildDirectory "generated_attention_session_bf16.cpp"
$bf16SessionText = [System.IO.File]::ReadAllText($sessionHarness).Replace(
    "attention-two-head-f64.cpp", "attention-two-head-bf16.cpp")
[System.IO.File]::WriteAllText($bf16SessionHarness, $bf16SessionText,
    [System.Text.UTF8Encoding]::new($false))
$bf16SessionExecutable = Join-Path $buildDirectory "generated_attention_session_bf16.exe"
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 `
    $bf16SessionHarness -o $bf16SessionExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated BF16 attention-session build failed" }
& $bf16SessionExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated BF16 attention-session checks failed" }
Write-Output "Generated BF16 session projected input rows and matched the exact identity fixture"

$causalAttentionModel = Join-Path $projectRoot "examples\causal_attention.proof"
$attentionPlan = Join-Path $buildDirectory "causal-attention.plan"
& $proofline plan $causalAttentionModel (Join-Path $projectRoot "examples\attention.cal") `
    0.05 $attentionPlan
if ($LASTEXITCODE -ne 0) { throw "Attention precision planning failed" }
$attentionDomain = Join-Path $buildDirectory "causal-attention.domain"
Set-Content -Path $attentionDomain -Value @("0 1", "0 1", "0 1", "0 1") -Encoding ascii
$attentionCertificate = Join-Path $buildDirectory "causal-attention-certificate.txt"
$attentionCertificateOutput = & $proofline certify $causalAttentionModel $attentionPlan `
    $attentionDomain 1.0 $attentionCertificate 2>&1
if ($LASTEXITCODE -ne 1 -or ($attentionCertificateOutput -join "`n") -notmatch "vector inputs only") {
    throw "The bounded-domain verifier did not clearly reject rank-two attention"
}
Write-Output "Bounded-domain certification clearly refuses rank-two attention until its bound is implemented"

$sequenceNormRopeModel = Join-Path $projectRoot "examples\sequence_norm_rope.proof"
$sequenceNormRopeSource = Join-Path $buildDirectory "sequence-norm-rope.cpp"
$sequenceNormRopeExecutable = Join-Path $buildDirectory "sequence-norm-rope.exe"
& $proofline compile $sequenceNormRopeModel $sequenceNormRopeSource
if ($LASTEXITCODE -ne 0) { throw "Sequence RMSNorm/RoPE code generation failed" }
& $compilerPath -std=c++20 -Wall -Wextra -Wpedantic -O2 `
    $sequenceNormRopeSource -o $sequenceNormRopeExecutable
if ($LASTEXITCODE -ne 0) { throw "Generated sequence RMSNorm/RoPE build failed" }
$primitiveInput = @("3", "4", "0", "0", "0", "2", "0", "0")
$primitiveGeneratedLine = & $sequenceNormRopeExecutable @primitiveInput
if ($LASTEXITCODE -ne 0) { throw "Generated sequence RMSNorm/RoPE execution failed" }
$primitiveGenerated = $primitiveGeneratedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$rowZeroRms = [Math]::Sqrt(6.25 + 1.0e-5)
$rowOneRms = [Math]::Sqrt(1.0 + 1.0e-5)
$sequencePrimitiveExpected = @(
    (3.0 / $rowZeroRms),
    (4.0 / $rowZeroRms),
    0.0,
    0.0,
    (-(2.0 / $rowOneRms) * [Math]::Sin(1.0)),
    ((2.0 / $rowOneRms) * [Math]::Cos(1.0)),
    0.0,
    0.0
)
for ($index = 0; $index -lt 8; $index++) {
    if ([Math]::Abs($primitiveGenerated[$index] - $sequencePrimitiveExpected[$index]) -gt 1.0e-14) {
        throw "Sequence RMSNorm/RoPE differs from the hand calculation at output $index"
    }
}
$primitiveRandom = [System.Random]::new(25211)
for ($sampleIndex = 0; $sampleIndex -lt 16; $sampleIndex++) {
    $sample = [System.Collections.Generic.List[string]]::new()
    for ($element = 0; $element -lt 8; $element++) {
        $sample.Add((($primitiveRandom.NextDouble() * 8.0) - 4.0).ToString(
            "R", [Globalization.CultureInfo]::InvariantCulture))
    }
    $primitiveReferenceLine = & $proofline run $sequenceNormRopeModel @sample
    if ($LASTEXITCODE -ne 0) { throw "Sequence RMSNorm/RoPE reference failed on sample $sampleIndex" }
    $primitiveGeneratedLine = & $sequenceNormRopeExecutable @sample
    if ($LASTEXITCODE -ne 0) { throw "Sequence RMSNorm/RoPE generated run failed on sample $sampleIndex" }
    $primitiveReference = (($primitiveReferenceLine -split ": ")[-1] -split '[,\s]+' |
        Where-Object { $_ }) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    $primitiveGenerated = $primitiveGeneratedLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
    for ($index = 0; $index -lt 8; $index++) {
        if ([Math]::Abs($primitiveReference[$index] - $primitiveGenerated[$index]) -gt 1.0e-14) {
            throw "Sequence RMSNorm/RoPE reference mismatch on sample $sampleIndex"
        }
    }
}
Write-Output "Sequence RMSNorm and position-aware RoPE matched hand calculations and 16 deterministic samples"
