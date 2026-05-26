param(
    [string]$Competition = "ariel-data-challenge-2025",
    [string]$DataRoot = "data",
    [string]$TrainPlanetId = "",
    [string]$TestPlanetId = ""
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force $DataRoot | Out-Null
New-Item -ItemType Directory -Force "outputs" | Out-Null

$fileListPath = "outputs\kaggle_files.txt"
kaggle competitions files -c $Competition | Out-File -Encoding utf8 $fileListPath
$fileList = Get-Content $fileListPath

if (-not $TrainPlanetId) {
    $trainMatch = $fileList | Select-String -Pattern "train/([^/\s]+)/AIRS-CH0_signal_0\.parquet" | Select-Object -First 1
    if (-not $trainMatch) { throw "Could not find a train AIRS signal file in Kaggle file list." }
    $TrainPlanetId = $trainMatch.Matches[0].Groups[1].Value
}

if (-not $TestPlanetId) {
    $testMatch = $fileList | Select-String -Pattern "test/([^/\s]+)/AIRS-CH0_signal_0\.parquet" | Select-Object -First 1
    if (-not $testMatch) { throw "Could not find a test AIRS signal file in Kaggle file list." }
    $TestPlanetId = $testMatch.Matches[0].Groups[1].Value
}

Write-Host "Selected train planet: $TrainPlanetId"
Write-Host "Selected test planet:  $TestPlanetId"

function Download-KaggleFile {
    param(
        [string]$RemotePath,
        [string]$LocalDir
    )
    New-Item -ItemType Directory -Force $LocalDir | Out-Null
    kaggle competitions download -c $Competition -f $RemotePath -p $LocalDir
}

$topLevelFiles = @(
    "adc_info.csv",
    "train.csv",
    "train_star_info.csv",
    "test_star_info.csv",
    "wavelengths.csv",
    "sample_submission.csv"
)

foreach ($file in $topLevelFiles) {
    Download-KaggleFile -RemotePath $file -LocalDir $DataRoot
}

$observationFiles = @(
    "AIRS-CH0_signal_0.parquet",
    "FGS1_signal_0.parquet",
    "AIRS-CH0_calibration_0/dark.parquet",
    "AIRS-CH0_calibration_0/dead.parquet",
    "AIRS-CH0_calibration_0/flat.parquet",
    "AIRS-CH0_calibration_0/linear_corr.parquet",
    "FGS1_calibration_0/dark.parquet",
    "FGS1_calibration_0/dead.parquet",
    "FGS1_calibration_0/flat.parquet",
    "FGS1_calibration_0/linear_corr.parquet"
)

foreach ($file in $observationFiles) {
    $localDir = Split-Path (Join-Path "$DataRoot\train\$TrainPlanetId" $file)
    Download-KaggleFile -RemotePath "train/$TrainPlanetId/$file" -LocalDir $localDir
}

foreach ($file in $observationFiles) {
    $localDir = Split-Path (Join-Path "$DataRoot\test\$TestPlanetId" $file)
    Download-KaggleFile -RemotePath "test/$TestPlanetId/$file" -LocalDir $localDir
}

Write-Host "Downloaded sample data under $DataRoot"
Write-Host "Train folder:"
Get-ChildItem "$DataRoot\train\$TrainPlanetId" -Recurse
Write-Host "Test folder:"
Get-ChildItem "$DataRoot\test\$TestPlanetId" -Recurse
