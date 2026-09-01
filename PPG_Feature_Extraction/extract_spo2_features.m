function [features, featureNames] = extract_spo2_features(spo2, Fs, nSeconds)
%EXTRACT_SPO2_FEATURES Extract one SpO2 feature row per second.
%
% INPUT
%   spo2       - raw SpO2 vector
%   Fs         - SpO2 sampling frequency (normally 1 Hz)
%   nSeconds   - number of labelled seconds
%
% OUTPUT
%   features      - nSeconds x 16 feature matrix
%   featureNames  - names of the 16 features

spo2 = double(spo2(:));

%% Settings

WindowLength = 5 * 60;       % 5 minute centred window
InvalidValue = 0;

DesaturationThreshold = 3;   % 3 percentage points
MinDesaturationDuration = 10;

%% Invalid SpO2 values

spo2(spo2 == InvalidValue) = NaN;

%% Feature names

featureNames = [ ...
    "spo2_mean"
    "spo2_median"
    "spo2_std"
    "spo2_min"
    "spo2_max"
    "spo2_range"
    "spo2_percent_below_90"
    "spo2_percent_below_92"
    "spo2_percent_below_95"
    "spo2_num_desaturations"
    "spo2_mean_desaturation_depth"
    "spo2_max_desaturation_depth"
    "spo2_total_desaturation_time"
    "spo2_mean_desaturation_duration"
    "spo2_max_desaturation_duration"
    "spo2_mean_recovery_time"
]';

features = nan(nSeconds, numel(featureNames));

%% Window size

halfWindow = floor(WindowLength * Fs / 2);

%% Extract one feature row per second

for t = 1:nSeconds

    % At 1 Hz, sample t corresponds to second t
    centre = round((t - 0.5) * Fs);
    centre = max(1, min(centre, numel(spo2)));

    windowStart = max(1, centre - halfWindow);
    windowEnd   = min(numel(spo2), centre + halfWindow - 1);

    windowSpO2 = spo2(windowStart:windowEnd);

    validSpO2 = windowSpO2(isfinite(windowSpO2));

    % Require at least 30 valid samples
    if numel(validSpO2) < 30
        continue;
    end

    %% Basic features

    meanSpO2   = mean(validSpO2);
    medianSpO2 = median(validSpO2);
    stdSpO2    = std(validSpO2);
    minSpO2    = min(validSpO2);
    maxSpO2    = max(validSpO2);
    rangeSpO2  = maxSpO2 - minSpO2;

    %% Threshold features

    percentBelow90 = 100 * mean(validSpO2 < 90);
    percentBelow92 = 100 * mean(validSpO2 < 92);
    percentBelow95 = 100 * mean(validSpO2 < 95);

    %% Desaturation features

    [numDesaturations, ...
     meanDepth, ...
     maxDepth, ...
     totalTime, ...
     meanDuration, ...
     maxDuration, ...
     meanRecovery] = ...
        analyseDesaturations( ...
            windowSpO2, ...
            Fs, ...
            DesaturationThreshold, ...
            MinDesaturationDuration);

    %% Store this second

    features(t,:) = [ ...
        meanSpO2 ...
        medianSpO2 ...
        stdSpO2 ...
        minSpO2 ...
        maxSpO2 ...
        rangeSpO2 ...
        percentBelow90 ...
        percentBelow92 ...
        percentBelow95 ...
        numDesaturations ...
        meanDepth ...
        maxDepth ...
        totalTime ...
        meanDuration ...
        maxDuration ...
        meanRecovery];

end

assert(size(features,1) == nSeconds);
assert(size(features,2) == numel(featureNames));

end


function [NumEvents, ...
          MeanDepth, ...
          MaxDepth, ...
          TotalTime, ...
          MeanDuration, ...
          MaxDuration, ...
          MeanRecovery] = ...
          analyseDesaturations(spo2, Fs, threshold, minDuration)

%ANALYSEDESATURATIONS Find and summarise SpO2 desaturation events.

%% Default outputs

NumEvents = 0;
MeanDepth = 0;
MaxDepth = 0;
TotalTime = 0;
MeanDuration = 0;
MaxDuration = 0;
MeanRecovery = 0;

%% Not enough data

if numel(spo2) < minDuration * Fs
    return;
end

%% Light smoothing

smoothWindow = max(1, round(3 * Fs));

smoothSpO2 = movmedian( ...
    spo2, ...
    smoothWindow, ...
    'omitmissing');

%% Recent SpO2 baseline

baselineWindow = max(1, round(60 * Fs));

% Look BACK over approximately 60 seconds
baseline = movmax( ...
    smoothSpO2, ...
    [baselineWindow-1 0], ...
    'omitmissing');

%% Drop from baseline

depth = baseline - smoothSpO2;

isDesaturation = depth >= threshold;

%% Find continuous desaturation regions

changes = diff([false; isDesaturation(:); false]);

startIndices = find(changes == 1);
endIndices   = find(changes == -1) - 1;

%% Store individual event information

Depths = [];
Durations = [];
RecoveryTimes = [];

%% Process each possible event

for k = 1:numel(startIndices)

    startIndex = startIndices(k);
    endIndex   = endIndices(k);

    %% Duration

    duration = ...
        (endIndex - startIndex + 1) / Fs;

    if duration < minDuration
        continue;
    end

    %% Baseline before event

    baselineStart = ...
        max(1, startIndex - baselineWindow);

    preBaseline = max( ...
        smoothSpO2(baselineStart:startIndex), ...
        [], ...
        'omitmissing');

    %% Find lowest SpO2 during event

    troughValue = min( ...
        smoothSpO2(startIndex:endIndex), ...
        [], ...
        'omitmissing');

    %% Desaturation depth

    eventDepth = preBaseline - troughValue;

    if eventDepth < threshold
        continue;
    end

    %% Store event

    Depths(end+1) = eventDepth;
    Durations(end+1) = duration;

    %% Recovery time

    recovery = NaN;

    % Recovery = return to within 1 percentage point
    % of the pre-desaturation baseline
    recoveryThreshold = 1;

    for r = endIndex+1:numel(smoothSpO2)

        if smoothSpO2(r) >= ...
                preBaseline - recoveryThreshold

            recovery = ...
                (r - endIndex) / Fs;

            break;
        end

    end

    if ~isnan(recovery)
        RecoveryTimes(end+1) = recovery;
    end

end

%% Summarise events

NumEvents = numel(Depths);

if NumEvents == 0
    return;
end

MeanDepth = mean(Depths);
MaxDepth = max(Depths);

TotalTime = sum(Durations);
MeanDuration = mean(Durations);
MaxDuration = max(Durations);

if ~isempty(RecoveryTimes)
    MeanRecovery = mean(RecoveryTimes);
end

end