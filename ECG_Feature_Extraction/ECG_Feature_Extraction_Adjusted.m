%% ECG Feature Extraction for Apnoea Classification
%
% Corrected + Optimised version
%
% Dataset:
%   ApnoeaECG.mat
%
% Inputs:
%   ECG   - ECG signal, 200 Hz
%   QRS   - approximate QRS detections (sample indices)
%   Class - 1 Hz class labels ('N' or 'A')
%
% Output:
%   125 ECG/HRV/EDR/RSA features per 60-second epoch
%
% Major corrections:
%   - Preserve invalid RR gaps for successive-difference metrics
%   - Use actual RR timestamps
%   - RR timestamps correspond to RR interval midpoints
%   - Nonlinear HRV uses contiguous valid RR data
%   - Limit nonlinear HRV sample count for runtime
%   - Properly scaled one-sided PSD
%   - HRV power fractions constrained to appropriate bands
%   - HRV dominant frequency restricted to HRV band
%   - Robust R/S ratio using a physiological post-R search region
%   - R amplitude measured at corrected R peak
%   - Morphology precomputed once per record
%   - Additional feature sanity checks
%   - Removed non-functioning PermutationEntropy feature
%   - Removed unstable DFA_Alpha1 and DFA_Alpha2 features
%   - Corrected EDR-Area feature name/order mismatch
%   - Constrained HRV respiratory power fraction to [0,1]
%   - R-wave amplitude uses magnitude to avoid lead-polarity sign effects
%
% -------------------------------------------------------------------------

%clear;
%clc;
%close all;

%% ========================================================
% LOAD DATA
% ========================================================

%load('TrainData.mat');

%% ========================================================
% PARAMETERS
% ========================================================

SR_ECG = 200;
SR_SpO2 = 1;

EpochLength = 60;
WindowLength = 5*60;

% ---------------------------------------------------------
% QRS correction
% ---------------------------------------------------------

QRS_SearchSamples = 40;
QRS_BaselineSamples = 20;

% ---------------------------------------------------------
% RR limits
% ---------------------------------------------------------

MinRR = 0.30;
MaxRR = 2.50;

% ---------------------------------------------------------
% HRV interpolation
% ---------------------------------------------------------

HRVFs = 4;

% ---------------------------------------------------------
% HRV frequency bands
% ---------------------------------------------------------

VLF_Low = 0.0033;
VLF_High = 0.04;

LF_Low = 0.04;
LF_High = 0.15;

HF_Low = 0.15;
HF_High = 0.40;

% ---------------------------------------------------------
% Respiratory frequency range
% ---------------------------------------------------------

RespLow = 0.10;
RespHigh = 0.50;

% ---------------------------------------------------------
% Nonlinear HRV runtime control
%
% Sample entropy and approximate entropy are O(N^2).
% Limiting the input length prevents these calculations
% from dominating the total runtime.
%
% Set to Inf to use all available contiguous RR data.
% ---------------------------------------------------------

MaxEntropyRR = 120;

% ---------------------------------------------------------
% R/S ratio parameters
% ---------------------------------------------------------

% Search for S wave approximately 10-60 ms after R.
RS_SearchStart = round(0.010*SR_ECG);
RS_SearchEnd   = round(0.060*SR_ECG);

% S wave must have at least this fraction of the maximum
% absolute deflection in the beat to be considered reliable.
RS_MinRelativeAmplitude = 0.05;

%% ========================================================
% FEATURE NAMES
% ========================================================

FeatureNames = { ...

    % TIME DOMAIN HRV - 25
    'MeanRR'
    'MedianRR'
    'SDNN'
    'RMSSD'
    'SDSD'
    'NN20'
    'pNN20'
    'NN30'
    'pNN30'
    'NN50'
    'pNN50'
    'MinRR'
    'MaxRR'
    'RangeRR'
    'CVRR'
    'IQR_RR'
    'MADR_RR'
    'MeanHR'
    'MedianHR'
    'SDHR'
    'MinHR'
    'MaxHR'
    'RangeHR'
    'CVHR'
    'MAD_HR'

    % FREQUENCY DOMAIN HRV - 18
    'VLFPower'
    'LFPower'
    'HFPower'
    'TotalHRVPower'
    'LF_HF_Ratio'
    'LF_Normalised'
    'HF_Normalised'
    'DominantHRVFrequency'
    'DominantHRVPower'
    'LF_PeakFrequency'
    'HF_PeakFrequency'
    'LF_HF_Power'
    'HF_PowerFraction'
    'HRVSpectralEntropy'
    'HRVSpectralCentroid'
    'HRVSpectralBandwidth'
    'RespiratoryPower_HRV'
    'RespiratoryPowerFraction_HRV'

    % POINCARE - 5
    'Poincare_SD1'
    'Poincare_SD2'
    'SD1_SD2'
    'PoincareArea'
    'SD1_SDNN'

    % NONLINEAR HRV - 2
    'SampleEntropy'
    'ApproximateEntropy'

    % HEART RATE DYNAMICS - 6
    'AccelerationFraction'
    'DecelerationFraction'
    'AccelerationMean'
    'DecelerationMean'
    'AccelerationDecelerationRatio'
    'TurningPointRatio'

    % ECG MORPHOLOGY - 25
    'MeanRAmplitude'
    'SDRAmplitude'
    'MedianRAmplitude'
    'MinRAmplitude'
    'MaxRAmplitude'
    'RangeRAmplitude'
    'MeanQRSAmplitude'
    'SDQRSAmplitude'
    'MedianQRSAmplitude'
    'MeanQRSEnergy'
    'SDQRSEnergy'
    'MeanQRSArea'
    'SDQRSArea'
    'MeanQRSWidth'
    'SDQRSWidth'
    'MeanMaxSlope'
    'SDMaxSlope'
    'MeanMinSlope'
    'SDMinSlope'
    'MeanRSRatio'
    'SDRSRatio'
    'ECGStd'
    'ECGRMS'
    'ECGRange'
    'ECGKurtosis'

    % MORPHOLOGY VARIABILITY - 5
    'RAmplitudeCV'
    'QRSAmplitudeCV'
    'QRSWidthCV'
    'QRSAreaCV'
    'QRSEnergyCV'

    % EDR-R - 10
    'EDR_R_Mean'
    'EDR_R_SD'
    'EDR_R_Range'
    'EDR_R_RespiratoryPower'
    'EDR_R_DominantFrequency'
    'EDR_R_RespiratoryRate'
    'EDR_R_DominantPower'
    'EDR_R_SpectralEntropy'
    'EDR_R_RelativeRespPower'
    'EDR_R_ZeroCrossingRate'

    % EDR-QRS - 6
    'EDR_QRS_RespiratoryPower'
    'EDR_QRS_DominantFrequency'
    'EDR_QRS_RespiratoryRate'
    'EDR_QRS_DominantPower'
    'EDR_QRS_SpectralEntropy'
    'EDR_QRS_RelativeRespPower'

    % EDR-ENERGY - 6
    'EDR_Energy_RespiratoryPower'
    'EDR_Energy_DominantFrequency'
    'EDR_Energy_RespiratoryRate'
    'EDR_Energy_DominantPower'
    'EDR_Energy_SpectralEntropy'
    'EDR_Energy_RelativeRespPower'

    % EDR-AREA - 6
    'EDR_Area_RespiratoryPower'
    'EDR_Area_DominantFrequency'
    'EDR_Area_RespiratoryRate'
    'EDR_Area_DominantPower'
    'EDR_Area_SpectralEntropy'
    'EDR_Area_RelativeRespPower'

    % EDR-SLOPE - 6
    'EDR_Slope_RespiratoryPower'
    'EDR_Slope_DominantFrequency'
    'EDR_Slope_RespiratoryRate'
    'EDR_Slope_DominantPower'
    'EDR_Slope_SpectralEntropy'
    'EDR_Slope_RelativeRespPower'

    % RSA - 5
    'RespiratoryPower_RSA'
    'DominantRespFrequency_RSA'
    'RespiratoryRate_RSA'
    'DominantRespPower_RSA'
    'RespiratorySpectralEntropy'
    };

NumFeatures = numel(FeatureNames);

if NumFeatures ~= 125
    error('FeatureNames contains %d features, expected 125.', ...
        NumFeatures);
end

%% ========================================================
% DATA SIZE
% ========================================================

NumRecords = numel(ECG);

fprintf('\n');
fprintf('============================================\n');
fprintf('ECG FEATURE EXTRACTION\n');
fprintf('============================================\n');
fprintf('Records: %d\n',NumRecords);
fprintf('Features: %d\n',NumFeatures);
fprintf('Epoch length: %d s\n',EpochLength);
fprintf('Feature window: %d s\n',WindowLength);
fprintf('Max nonlinear RR samples: %g\n',MaxEntropyRR);
fprintf('============================================\n\n');

%% ========================================================
% OUTPUT CONTAINERS
% ========================================================

X_ECG = cell(NumRecords,1);
T_ECG = cell(NumRecords,1);
EpochTime_ECG = cell(NumRecords,1);

CorrectedRPeaks = cell(NumRecords,1);
RR_All = cell(NumRecords,1);

totalTimer = tic;

%% ========================================================
% PROCESS RECORDS
% ========================================================

for record = 1:NumRecords

    recordTimer = tic;

    fprintf('Record %3d/%3d ... ',record,NumRecords);

    %% -----------------------------------------------------
    % LOAD RECORD
    % -----------------------------------------------------

    ecg = double(ECG{record}(:));
    qrs = double(QRS{record}(:));

    classLabels = Class{record};
    classLabels = classLabels(:);

    NumSamples = numel(ecg);

    %% =====================================================
    % QRS CORRECTION
    % =====================================================

    NumQRS = numel(qrs);

    RPeaks = nan(NumQRS,1);

    for k = 1:NumQRS

        qrsIndex = round(qrs(k));

        if qrsIndex < 1 || qrsIndex >= NumSamples
            continue;
        end

        baselineStart = max(1, ...
            qrsIndex-QRS_BaselineSamples);

        baseline = median( ...
            ecg(baselineStart:qrsIndex));

        searchStart = qrsIndex+1;

        searchEnd = min( ...
            qrsIndex+QRS_SearchSamples, ...
            NumSamples);

        if searchStart > searchEnd
            continue;
        end

        segment = ...
            ecg(searchStart:searchEnd)-baseline;

        [~,localIndex] = max(abs(segment));

        RPeaks(k) = ...
            searchStart+localIndex-1;

    end

    RPeaks = RPeaks(isfinite(RPeaks));

    % ------------------------------------------------------
    % Remove very close duplicate detections
    % ------------------------------------------------------

    if numel(RPeaks) > 1

        keep = true(size(RPeaks));

        for k = 2:numel(RPeaks)

            if RPeaks(k)-RPeaks(k-1) <= ...
                    round(0.25*SR_ECG)

                keep(k) = false;

            end

        end

        RPeaks = RPeaks(keep);

    end

    CorrectedRPeaks{record} = RPeaks;

    %% =====================================================
    % RR INTERVALS
    % =====================================================

    RR = diff(RPeaks)/SR_ECG;

    RR(RR < MinRR | RR > MaxRR) = NaN;

    RR_All{record} = RR;

    % ------------------------------------------------------
    % Actual RR timestamps
    %
    % Each RR interval is associated with its midpoint.
    % This is preferable to using only the ending R peak.
    % ------------------------------------------------------

    if numel(RPeaks) >= 2

        RRTime = ...
            ((RPeaks(1:end-1)+RPeaks(2:end))/2 - 1) ...
            /SR_ECG;

    else

        RRTime = [];

    end

    %% =====================================================
    % PRECOMPUTE MORPHOLOGY FOR ENTIRE RECORD
    % =====================================================

    NumBeatsTotal = numel(RPeaks);

    MorphR = nan(NumBeatsTotal,1);
    MorphQRSamp = nan(NumBeatsTotal,1);
    MorphEnergy = nan(NumBeatsTotal,1);
    MorphArea = nan(NumBeatsTotal,1);
    MorphWidth = nan(NumBeatsTotal,1);
    MorphMaxSlope = nan(NumBeatsTotal,1);
    MorphMinSlope = nan(NumBeatsTotal,1);
    MorphRSRatio = nan(NumBeatsTotal,1);
    MorphBeatTimes = nan(NumBeatsTotal,1);

    MorphBefore = round(0.04*SR_ECG);
    MorphAfter = round(0.08*SR_ECG);

    for k = 1:NumBeatsTotal

        r = RPeaks(k);

        startIdx = max(1,r-MorphBefore);
        endIdx = min(NumSamples,r+MorphAfter);

        beat = ecg(startIdx:endIdx);

        if numel(beat) < 5
            continue;
        end

        beat = beat(:);

        %% -------------------------------------------------
        % LOCAL BASELINE
        % -------------------------------------------------

        baselineSamples = min( ...
            max(2,round(0.15*numel(beat))), ...
            numel(beat));

        baseline = median( ...
            beat(1:baselineSamples));

        beatCorrected = beat-baseline;

        %% -------------------------------------------------
        % R AMPLITUDE
        %
        % Use the corrected R-peak location directly.
        % -------------------------------------------------

        rLocal = r-startIdx+1;

        if rLocal >= 1 && ...
                rLocal <= numel(beatCorrected)

            rValue = ...
                abs(beatCorrected(rLocal));

        else

            rValue = NaN;

        end

        %% -------------------------------------------------
        % QRS AMPLITUDE
        % -------------------------------------------------

        qrsAmp = ...
            max(beatCorrected)- ...
            min(beatCorrected);

        %% -------------------------------------------------
        % QRS ENERGY
        % -------------------------------------------------

        energy = ...
            sum(beatCorrected.^2);

        %% -------------------------------------------------
        % QRS AREA
        % -------------------------------------------------

        areaValue = ...
            trapz(abs(beatCorrected));

        %% -------------------------------------------------
        % QRS SLOPE
        % -------------------------------------------------

        derivative = ...
            diff(beatCorrected);

        maxSlope = ...
            max(derivative);

        minSlope = ...
            min(derivative);

        %% -------------------------------------------------
        % QRS WIDTH
        % -------------------------------------------------

        halfAmp = ...
            0.5*max(abs(beatCorrected));

        aboveHalf = ...
            abs(beatCorrected) >= halfAmp;

        if any(aboveHalf)

            firstHalf = ...
                find(aboveHalf,1,'first');

            lastHalf = ...
                find(aboveHalf,1,'last');

            width = ...
                (lastHalf-firstHalf+1)/SR_ECG;

        else

            width = NaN;

        end

        %% -------------------------------------------------
        % ROBUST R/S RATIO
        %
        % Search only after the R peak, where the S wave
        % would normally occur.
        %
        % A very small negative deflection is treated as
        % "no reliable S wave" rather than producing an
        % enormous ratio.
        % -------------------------------------------------

        RSratio = NaN;

        if rLocal >= 1 && ...
                rLocal <= numel(beatCorrected)

            searchStart = ...
                rLocal + RS_SearchStart;

            searchEnd = ...
                min(rLocal + RS_SearchEnd, ...
                numel(beatCorrected));

            if searchStart <= searchEnd

                postR = ...
                    beatCorrected(searchStart:searchEnd);

                sValue = ...
                    min(postR);

                sAmplitude = ...
                    abs(sValue);

                maxBeatAmplitude = ...
                    max(abs(beatCorrected));

                minimumSAmplitude = ...
                    RS_MinRelativeAmplitude * ...
                    maxBeatAmplitude;

                if sAmplitude >= minimumSAmplitude && ...
                        sAmplitude > 0

                    RSratio = ...
                        abs(rValue)/sAmplitude;

                end

            end

        end

        %% -------------------------------------------------
        % STORE MORPHOLOGY
        % -------------------------------------------------

        MorphR(k) = rValue;
        MorphQRSamp(k) = qrsAmp;
        MorphEnergy(k) = energy;
        MorphArea(k) = areaValue;
        MorphWidth(k) = width;
        MorphMaxSlope(k) = maxSlope;
        MorphMinSlope(k) = minSlope;
        MorphRSRatio(k) = RSratio;

        MorphBeatTimes(k) = ...
            (r-1)/SR_ECG;

    end

    %% =====================================================
    % EPOCH SETUP
    % =====================================================

    NumEpochs = ...
        floor(numel(classLabels)/EpochLength);

    X = nan(NumEpochs,NumFeatures);

    T = strings(NumEpochs,1);

    EpochTime = nan(NumEpochs,1);

    %% =====================================================
    % EPOCH LOOP
    % =====================================================

    for epoch = 1:NumEpochs

        %% -------------------------------------------------
        % CLASS LABEL
        % -------------------------------------------------

        epochStart = ...
            (epoch-1)*EpochLength+1;

        epochEnd = ...
            epoch*EpochLength;

        labels = ...
            classLabels(epochStart:epochEnd);

        if any(labels == 'A')

            T(epoch) = "A";

        elseif any(labels == 'N')

            T(epoch) = "N";

        else

            T(epoch) = "N";

        end

        EpochTime(epoch) = ...
            ((epochStart+epochEnd)/2)-1;

        %% -------------------------------------------------
        % CENTRED 5-MINUTE WINDOW
        % -------------------------------------------------

        centreTime = ...
            (epochStart+epochEnd)/2;

        windowStartTime = ...
            max(1, ...
            centreTime-WindowLength/2);

        windowEndTime = ...
            min(numel(classLabels), ...
            centreTime+WindowLength/2);

        %% -------------------------------------------------
        % ECG WINDOW
        % -------------------------------------------------

        ecgStart = ...
            round((windowStartTime-1)*SR_ECG)+1;

        ecgEnd = ...
            min(NumSamples, ...
            round(windowEndTime*SR_ECG));

        if ecgStart >= ecgEnd

            continue;

        end

        ecgWindow = ...
            ecg(ecgStart:ecgEnd);

        %% -------------------------------------------------
        % BEATS IN WINDOW
        % -------------------------------------------------

        beatMask = ...
            MorphBeatTimes >= windowStartTime & ...
            MorphBeatTimes <= windowEndTime;

        windowR = ...
            MorphR(beatMask);

        windowQRSamp = ...
            MorphQRSamp(beatMask);

        windowEnergy = ...
            MorphEnergy(beatMask);

        windowArea = ...
            MorphArea(beatMask);

        windowWidth = ...
            MorphWidth(beatMask);

        windowMaxSlope = ...
            MorphMaxSlope(beatMask);

        windowMinSlope = ...
            MorphMinSlope(beatMask);

        windowRSRatio = ...
            MorphRSRatio(beatMask);

        windowBeatTimes = ...
            MorphBeatTimes(beatMask);

        %% -------------------------------------------------
        % RR IN WINDOW
        %
        % IMPORTANT:
        % Keep NaNs in rrWindow.
        % This prevents invalid RR intervals from being
        % incorrectly treated as adjacent to the next
        % valid interval.
        % -------------------------------------------------

        rrMask = ...
            RRTime >= windowStartTime & ...
            RRTime <= windowEndTime;

        rrWindow = ...
            RR(rrMask);

        rrWindow = rrWindow(:);

        rrWindowTime = ...
            RRTime(rrMask);

        rrWindowTime = ...
            rrWindowTime(:);

        validRR = ...
            rrWindow(isfinite(rrWindow));

        %% -------------------------------------------------
        % Initialise successive RR differences
        % -------------------------------------------------

        dRR = [];

        if numel(rrWindow) >= 2

            validPairMask = ...
                isfinite(rrWindow(1:end-1)) & ...
                isfinite(rrWindow(2:end));

            dRR_all = ...
                diff(rrWindow);

            dRR = ...
                dRR_all(validPairMask);

        end

        %% =================================================
        % FEATURE VECTOR
        % =================================================

        values = ...
            nan(1,NumFeatures);

        f = 0;

        %% =================================================
        % 1. TIME DOMAIN HRV
        % =================================================

        if numel(validRR) >= 2

            meanRR = ...
                mean(validRR);

            medianRR = ...
                median(validRR);

            SDNN = ...
                std(validRR,0);

            if ~isempty(dRR)

                RMSSD = ...
                    sqrt(mean(dRR.^2));

                SDSD = ...
                    std(dRR,0);

                NN20 = ...
                    sum(abs(dRR) > 0.020);

                NN30 = ...
                    sum(abs(dRR) > 0.030);

                NN50 = ...
                    sum(abs(dRR) > 0.050);

                pNN20 = ...
                    100*NN20/numel(dRR);

                pNN30 = ...
                    100*NN30/numel(dRR);

                pNN50 = ...
                    100*NN50/numel(dRR);

            else

                [RMSSD,SDSD, ...
                    NN20,pNN20, ...
                    NN30,pNN30, ...
                    NN50,pNN50] = ...
                    deal(NaN);

            end

            minRR = ...
                min(validRR);

            maxRR = ...
                max(validRR);

            rangeRR = ...
                maxRR-minRR;

            CVRR = ...
                safeRatio(SDNN,abs(meanRR));

            IQR_RR = ...
                percentile_fast(validRR,75)- ...
                percentile_fast(validRR,25);

            MADR_RR = ...
                median(abs(validRR-medianRR));

            HR = ...
                60./validRR;

            meanHR = ...
                mean(HR);

            medianHR = ...
                median(HR);

            SDHR = ...
                std(HR,0);

            minHR = ...
                min(HR);

            maxHR = ...
                max(HR);

            rangeHR = ...
                maxHR-minHR;

            CVHR = ...
                safeRatio(SDHR,abs(meanHR));

            MAD_HR = ...
                median(abs(HR-medianHR));

        else

            [meanRR,medianRR,SDNN, ...
                RMSSD,SDSD, ...
                NN20,pNN20, ...
                NN30,pNN30, ...
                NN50,pNN50, ...
                minRR,maxRR,rangeRR, ...
                CVRR,IQR_RR,MADR_RR, ...
                meanHR,medianHR,SDHR, ...
                minHR,maxHR,rangeHR, ...
                CVHR,MAD_HR] = ...
                deal(NaN);

        end

        block = [ ...
            meanRR
            medianRR
            SDNN
            RMSSD
            SDSD
            NN20
            pNN20
            NN30
            pNN30
            NN50
            pNN50
            minRR
            maxRR
            rangeRR
            CVRR
            IQR_RR
            MADR_RR
            meanHR
            medianHR
            SDHR
            minHR
            maxHR
            rangeHR
            CVHR
            MAD_HR];

        values(f+1:f+25) = ...
            block(:)';

        f = f+25;

        %% =================================================
        % 2. FREQUENCY DOMAIN HRV
        % =================================================

        [ ...
            VLFPower, ...
            LFPower, ...
            HFPower, ...
            TotalHRVPower, ...
            LF_HF_Ratio, ...
            LF_Normalised, ...
            HF_Normalised, ...
            DominantHRVFrequency, ...
            DominantHRVPower, ...
            LF_PeakFrequency, ...
            HF_PeakFrequency, ...
            LF_HF_Power, ...
            HF_PowerFraction, ...
            HRVSpectralEntropy, ...
            HRVSpectralCentroid, ...
            HRVSpectralBandwidth, ...
            RespiratoryPower_HRV, ...
            RespiratoryPowerFraction_HRV] = ...
            calculateHRVFrequencyFeatures( ...
            rrWindow, ...
            rrWindowTime, ...
            HRVFs, ...
            VLF_Low,VLF_High, ...
            LF_Low,LF_High, ...
            HF_Low,HF_High, ...
            RespLow,RespHigh);

        block = [ ...
            VLFPower
            LFPower
            HFPower
            TotalHRVPower
            LF_HF_Ratio
            LF_Normalised
            HF_Normalised
            DominantHRVFrequency
            DominantHRVPower
            LF_PeakFrequency
            HF_PeakFrequency
            LF_HF_Power
            HF_PowerFraction
            HRVSpectralEntropy
            HRVSpectralCentroid
            HRVSpectralBandwidth
            RespiratoryPower_HRV
            RespiratoryPowerFraction_HRV];

        values(f+1:f+18) = ...
            block(:)';

        f = f+18;

        %% =================================================
        % 3. POINCARE
        % =================================================

        if numel(validRR) >= 3 && ...
                numel(dRR) >= 2

            SD1 = ...
                sqrt(0.5)*std(dRR,0);

            SD2 = ...
                sqrt(max(0, ...
                2*SDNN^2-SD1^2));

            SD1_SD2 = ...
                safeRatio(SD1,SD2);

            PoincareArea = ...
                pi*SD1*SD2;

            SD1_SDNN = ...
                safeRatio(SD1,SDNN);

        else

            [SD1,SD2, ...
                SD1_SD2,PoincareArea, ...
                SD1_SDNN] = ...
                deal(NaN);

        end

        block = [ ...
            SD1
            SD2
            SD1_SD2
            PoincareArea
            SD1_SDNN];

        values(f+1:f+5) = ...
            block(:)';

        f = f+5;

        %% =================================================
        % 4. NONLINEAR HRV
        % =================================================
        %
        % Use the longest contiguous valid RR sequence.
        % This prevents entropy/DFA calculations from
        % artificially connecting RR values across invalid
        % detections.
        % =================================================

        nonlinearRR = ...
            longestContiguousSegment(rrWindow);

        nonlinearRR = ...
            limitRRLength(nonlinearRR,MaxEntropyRR);

        if numel(nonlinearRR) >= 20

            SampleEntropyValue = ...
                sampleEntropyFast(nonlinearRR);

            ApproximateEntropyValue = ...
                approximateEntropyFast(nonlinearRR);


        else

            [SampleEntropyValue, ...
                ApproximateEntropyValue] = ...
                deal(NaN);

        end

        block = [ ...
            SampleEntropyValue
            ApproximateEntropyValue];

        values(f+1:f+2) = ...
            block(:)';

        f = f+2;

        %% =================================================
        % 5. HEART RATE DYNAMICS
        % =================================================

        if numel(dRR) >= 2

            acceleration = ...
                dRR < 0;

            deceleration = ...
                dRR > 0;

            AccelerationFraction = ...
                mean(acceleration);

            DecelerationFraction = ...
                mean(deceleration);

            if any(acceleration)

                AccelerationMean = ...
                    mean(abs(dRR(acceleration)));

            else

                AccelerationMean = NaN;

            end

            if any(deceleration)

                DecelerationMean = ...
                    mean(abs(dRR(deceleration)));

            else

                DecelerationMean = NaN;

            end

            AccelerationDecelerationRatio = ...
                safeRatio( ...
                AccelerationMean, ...
                DecelerationMean);

            TurningPointRatio = ...
                turningPointRatioFast(rrWindow);

        else

            [AccelerationFraction, ...
                DecelerationFraction, ...
                AccelerationMean, ...
                DecelerationMean, ...
                AccelerationDecelerationRatio, ...
                TurningPointRatio] = ...
                deal(NaN);

        end

        block = [ ...
            AccelerationFraction
            DecelerationFraction
            AccelerationMean
            DecelerationMean
            AccelerationDecelerationRatio
            TurningPointRatio];

        values(f+1:f+6) = ...
            block(:)';

        f = f+6;

        %% =================================================
        % 6. ECG MORPHOLOGY
        % =================================================

        if isempty(windowR)

            [MeanRAmplitude, ...
                SDRAmplitude, ...
                MedianRAmplitude, ...
                MinRAmplitude, ...
                MaxRAmplitude, ...
                RangeRAmplitude, ...
                MeanQRSAmplitude, ...
                SDQRSAmplitude, ...
                MedianQRSAmplitude, ...
                MeanQRSEnergy, ...
                SDQRSEnergy, ...
                MeanQRSArea, ...
                SDQRSArea, ...
                MeanQRSWidth, ...
                SDQRSWidth, ...
                MeanMaxSlope, ...
                SDMaxSlope, ...
                MeanMinSlope, ...
                SDMinSlope, ...
                MeanRSRatio, ...
                SDRSRatio] = ...
                deal(NaN);

        else

            MeanRAmplitude = ...
                mean(windowR,'omitnan');

            SDRAmplitude = ...
                std(windowR,0,'omitnan');

            MedianRAmplitude = ...
                median(windowR,'omitnan');

            MinRAmplitude = ...
                min(windowR,[],'omitnan');

            MaxRAmplitude = ...
                max(windowR,[],'omitnan');

            RangeRAmplitude = ...
                MaxRAmplitude-MinRAmplitude;

            MeanQRSAmplitude = ...
                mean(windowQRSamp,'omitnan');

            SDQRSAmplitude = ...
                std(windowQRSamp,0,'omitnan');

            MedianQRSAmplitude = ...
                median(windowQRSamp,'omitnan');

            MeanQRSEnergy = ...
                mean(windowEnergy,'omitnan');

            SDQRSEnergy = ...
                std(windowEnergy,0,'omitnan');

            MeanQRSArea = ...
                mean(windowArea,'omitnan');

            SDQRSArea = ...
                std(windowArea,0,'omitnan');

            MeanQRSWidth = ...
                mean(windowWidth,'omitnan');

            SDQRSWidth = ...
                std(windowWidth,0,'omitnan');

            MeanMaxSlope = ...
                mean(windowMaxSlope,'omitnan');

            SDMaxSlope = ...
                std(windowMaxSlope,0,'omitnan');

            MeanMinSlope = ...
                mean(windowMinSlope,'omitnan');

            SDMinSlope = ...
                std(windowMinSlope,0,'omitnan');

            MeanRSRatio = ...
                mean(windowRSRatio,'omitnan');

            SDRSRatio = ...
                std(windowRSRatio,0,'omitnan');

        end

        %% -------------------------------------------------
        % Whole ECG window statistics
        % -------------------------------------------------

        ECGStd = ...
            std(ecgWindow,0);

        ECGRMS = ...
            sqrt(mean(ecgWindow.^2));

        ECGRange = ...
            max(ecgWindow)-min(ecgWindow);

        ECGKurtosis = ...
            kurtosisManualFast(ecgWindow);

        block = [ ...
            MeanRAmplitude
            SDRAmplitude
            MedianRAmplitude
            MinRAmplitude
            MaxRAmplitude
            RangeRAmplitude
            MeanQRSAmplitude
            SDQRSAmplitude
            MedianQRSAmplitude
            MeanQRSEnergy
            SDQRSEnergy
            MeanQRSArea
            SDQRSArea
            MeanQRSWidth
            SDQRSWidth
            MeanMaxSlope
            SDMaxSlope
            MeanMinSlope
            SDMinSlope
            MeanRSRatio
            SDRSRatio
            ECGStd
            ECGRMS
            ECGRange
            ECGKurtosis];

        values(f+1:f+25) = ...
            block(:)';

        f = f+25;

        %% =================================================
        % 7. MORPHOLOGY VARIABILITY
        % =================================================

        RAmplitudeCV = ...
            coefficientVariation(windowR);

        QRSAmplitudeCV = ...
            coefficientVariation(windowQRSamp);

        QRSWidthCV = ...
            coefficientVariation(windowWidth);

        QRSAreaCV = ...
            coefficientVariation(windowArea);

        QRSEnergyCV = ...
            coefficientVariation(windowEnergy);

        block = [ ...
            RAmplitudeCV
            QRSAmplitudeCV
            QRSWidthCV
            QRSAreaCV
            QRSEnergyCV];

        values(f+1:f+5) = ...
            block(:)';

        f = f+5;

        %% =================================================
        % 8. EDR-R
        % =================================================

        EDR_R_features = ...
            edrFeaturesFast( ...
            windowBeatTimes, ...
            windowR, ...
            RespLow, ...
            RespHigh);

        values(f+1:f+10) = ...
            EDR_R_features;

        f = f+10;

        %% =================================================
        % 9. EDR-QRS
        % =================================================

        EDR_QRS_features = ...
            edrFeaturesBasicFast( ...
            windowBeatTimes, ...
            windowQRSamp, ...
            RespLow, ...
            RespHigh);

        values(f+1:f+6) = ...
            EDR_QRS_features;

        f = f+6;

        %% =================================================
        % 10. EDR-ENERGY
        % =================================================

        EDR_Energy_features = ...
            edrFeaturesBasicFast( ...
            windowBeatTimes, ...
            windowEnergy, ...
            RespLow, ...
            RespHigh);

        values(f+1:f+6) = ...
            EDR_Energy_features;

        f = f+6;

        %% =================================================
        % 11. EDR-AREA
        % =================================================

        EDR_Area_features = ...
            edrFeaturesBasicFast( ...
            windowBeatTimes, ...
            windowArea, ...
            RespLow, ...
            RespHigh);

        values(f+1:f+6) = ...
            EDR_Area_features;

        f = f+6;

        %% =================================================
        % 12. EDR-SLOPE
        % =================================================

        EDR_Slope_features = ...
            edrFeaturesBasicFast( ...
            windowBeatTimes, ...
            windowMaxSlope, ...
            RespLow, ...
            RespHigh);

        values(f+1:f+6) = ...
            EDR_Slope_features;

        f = f+6;

        %% =================================================
        % 13. RSA
        % =================================================

        [RespPowerRSA, ...
            DominantRespFrequencyRSA, ...
            RespiratoryRateRSA, ...
            DominantPowerRSA, ...
            RespiratorySpectralEntropyRSA] = ...
            rsaFeaturesFast( ...
            rrWindow, ...
            rrWindowTime, ...
            HRVFs, ...
            RespLow, ...
            RespHigh);

        block = [ ...
            RespPowerRSA
            DominantRespFrequencyRSA
            RespiratoryRateRSA
            DominantPowerRSA
            RespiratorySpectralEntropyRSA];

        values(f+1:f+5) = ...
            block(:)';

        f = f+5;

        %% =================================================
        % CHECK FEATURE COUNT
        % =================================================

        if f ~= NumFeatures

            error( ...
                'Feature count mismatch: expected %d, generated %d.', ...
                NumFeatures,f);

        end

        X(epoch,:) = values;

    end

    %% =====================================================
    % SAVE RECORD
    % =====================================================

    X_ECG{record} = X;
    T_ECG{record} = T;
    EpochTime_ECG{record} = EpochTime;

    elapsed = toc(recordTimer);

    fprintf('%.1f s (%d epochs)\n', ...
        elapsed,NumEpochs);

end

%% ========================================================
% COMBINE RECORDS
% ========================================================

fprintf('\nCombining records...\n');

TotalEpochs = ...
    sum(cellfun(@(x)size(x,1),X_ECG));

X_all = ...
    nan(TotalEpochs,NumFeatures);

T_all = ...
    strings(TotalEpochs,1);

RecordID_all = ...
    nan(TotalEpochs,1);

EpochTime_all = ...
    nan(TotalEpochs,1);

row = 0;

for record = 1:NumRecords

    n = ...
        size(X_ECG{record},1);

    idx = ...
        row+1:row+n;

    X_all(idx,:) = ...
        X_ECG{record};

    T_all(idx) = ...
        T_ECG{record};

    RecordID_all(idx) = ...
        record;

    EpochTime_all(idx) = ...
        EpochTime_ECG{record};

    row = row+n;

end

%% ========================================================
% SUMMARY
% ========================================================

totalTime = ...
    toc(totalTimer);

fprintf('\n');
fprintf('============================================\n');
fprintf('ECG FEATURE EXTRACTION COMPLETE\n');
fprintf('============================================\n');

fprintf('Number of records: %d\n',NumRecords);
fprintf('Total epochs: %d\n',size(X_all,1));
fprintf('Number of features: %d\n',size(X_all,2));

fprintf('Normal epochs: %d\n', ...
    sum(T_all=="N"));

fprintf('Apnoea epochs: %d\n', ...
    sum(T_all=="A"));

fprintf('\nTotal runtime: %.2f minutes\n', ...
    totalTime/60);

fprintf('Average per record: %.2f seconds\n', ...
    totalTime/NumRecords);

%% ========================================================
% FEATURE SUMMARY
% ========================================================

fprintf('\n');
fprintf('FEATURE SUMMARY\n');
fprintf('--------------------------------------------\n');

for k = 1:NumFeatures

    featureMean = ...
        mean(X_all(:,k),'omitnan');

    featureStd = ...
        std(X_all(:,k),0,'omitnan');

    fprintf( ...
        '%-35s Mean = %10.4g   Std = %10.4g\n', ...
        FeatureNames{k}, ...
        featureMean, ...
        featureStd);

end

%% ========================================================
% NORMAL VS APNOEA
% ========================================================

fprintf('\n');
fprintf('NORMAL VS APNOEA\n');
fprintf('--------------------------------------------\n');

normalIdx = ...
    T_all=="N";

apnoeaIdx = ...
    T_all=="A";

for k = 1:NumFeatures

    normalMean = ...
        mean(X_all(normalIdx,k),'omitnan');

    apnoeaMean = ...
        mean(X_all(apnoeaIdx,k),'omitnan');

    fprintf( ...
        '%-35s N = %10.4g   A = %10.4g\n', ...
        FeatureNames{k}, ...
        normalMean, ...
        apnoeaMean);

end

%% ========================================================
% AUTOMATIC FEATURE SANITY CHECK
% ========================================================

fprintf('\n');
fprintf('FEATURE SANITY CHECK\n');
fprintf('--------------------------------------------\n');

for k = 1:NumFeatures

    x = X_all(:,k);

    finiteX = ...
        x(isfinite(x));

    if isempty(finiteX)

        fprintf( ...
            '%-35s ALL NaN/Inf\n', ...
            FeatureNames{k});

        continue;

    end

    nanCount = ...
        sum(isnan(x));

    infCount = ...
        sum(isinf(x));

    if nanCount > 0 || infCount > 0

        fprintf( ...
            '%-35s NaN = %d, Inf = %d\n', ...
            FeatureNames{k}, ...
            nanCount, ...
            infCount);

    end

end

%% ========================================================
% FRACTION RANGE CHECKS
% ========================================================

fractionFeatures = { ...
    'LF_Normalised'
    'HF_Normalised'
    'HF_PowerFraction'
    'RespiratoryPowerFraction_HRV'
    'EDR_R_RelativeRespPower'
    'EDR_QRS_RelativeRespPower'
    'EDR_Energy_RelativeRespPower'
    'EDR_Area_RelativeRespPower'
    'EDR_Slope_RelativeRespPower'};

fprintf('\n');
fprintf('FRACTION RANGE CHECKS\n');
fprintf('--------------------------------------------\n');

for i = 1:numel(fractionFeatures)

    idx = ...
        find(strcmp(FeatureNames, ...
        fractionFeatures{i}),1);

    if isempty(idx)
        continue;
    end

    x = ...
        X_all(:,idx);

    x = ...
        x(isfinite(x));

    if isempty(x)
        continue;
    end

    fprintf( ...
        '%-35s Min = %.4g   Max = %.4g\n', ...
        fractionFeatures{i}, ...
        min(x), ...
        max(x));

    if min(x) < -1e-6 || ...
            max(x) > 1+1e-6

        fprintf( ...
            'WARNING: %s contains values outside [0,1].\n', ...
            fractionFeatures{i});

    end

end

%% ========================================================
% HRV POWER CONSISTENCY CHECK
% ========================================================

idxVLF = ...
    find(strcmp(FeatureNames,'VLFPower'),1);

idxLF = ...
    find(strcmp(FeatureNames,'LFPower'),1);

idxHF = ...
    find(strcmp(FeatureNames,'HFPower'),1);

idxTotal = ...
    find(strcmp(FeatureNames,'TotalHRVPower'),1);

if ~isempty(idxVLF) && ...
        ~isempty(idxLF) && ...
        ~isempty(idxHF) && ...
        ~isempty(idxTotal)

    VLF = X_all(:,idxVLF);
    LF = X_all(:,idxLF);
    HF = X_all(:,idxHF);
    Total = X_all(:,idxTotal);

    validPower = ...
        isfinite(VLF) & ...
        isfinite(LF) & ...
        isfinite(HF) & ...
        isfinite(Total);

    if any(validPower)

        powerError = ...
            abs(Total(validPower) - ...
            (VLF(validPower)+ ...
            LF(validPower)+ ...
            HF(validPower)));

        maxPowerError = ...
            max(powerError);

        fprintf('\n');
        fprintf('HRV POWER CONSISTENCY CHECK\n');
        fprintf('Maximum |Total - (VLF+LF+HF)| = %.6g\n', ...
            maxPowerError);

        if maxPowerError > 1e-10

            fprintf( ...
                'WARNING: HRV power components are inconsistent.\n');

        else

            fprintf( ...
                'HRV power components are internally consistent.\n');

        end

    end

end

%% ========================================================
% SAVE
% ========================================================

save('ECG_Features.mat', ...
    'X_ECG', ...
    'T_ECG', ...
    'X_all', ...
    'T_all', ...
    'FeatureNames', ...
    'CorrectedRPeaks', ...
    'RR_All', ...
    'EpochTime_ECG', ...
    'RecordID_all', ...
    'EpochTime_all');

fprintf('\n');
fprintf('Results saved to ECG_Features.mat\n');

%% ========================================================
% HRV FREQUENCY FEATURES
% ========================================================

function [ ...
    VLFPower, ...
    LFPower, ...
    HFPower, ...
    TotalHRVPower, ...
    LF_HF_Ratio, ...
    LF_Normalised, ...
    HF_Normalised, ...
    DominantHRVFrequency, ...
    DominantHRVPower, ...
    LF_PeakFrequency, ...
    HF_PeakFrequency, ...
    LF_HF_Power, ...
    HF_PowerFraction, ...
    HRVSpectralEntropy, ...
    HRVSpectralCentroid, ...
    HRVSpectralBandwidth, ...
    RespiratoryPower_HRV, ...
    RespiratoryPowerFraction_HRV] = ...
    calculateHRVFrequencyFeatures( ...
    rrWindow, ...
    rrTime, ...
    Fs, ...
    VLF_Low,VLF_High, ...
    LF_Low,LF_High, ...
    HF_Low,HF_High, ...
    RespLow,RespHigh)

VLFPower = NaN;
LFPower = NaN;
HFPower = NaN;
TotalHRVPower = NaN;
LF_HF_Ratio = NaN;
LF_Normalised = NaN;
HF_Normalised = NaN;
DominantHRVFrequency = NaN;
DominantHRVPower = NaN;
LF_PeakFrequency = NaN;
HF_PeakFrequency = NaN;
LF_HF_Power = NaN;
HF_PowerFraction = NaN;
HRVSpectralEntropy = NaN;
HRVSpectralCentroid = NaN;
HRVSpectralBandwidth = NaN;
RespiratoryPower_HRV = NaN;
RespiratoryPowerFraction_HRV = NaN;

rrWindow = rrWindow(:);
rrTime = rrTime(:);

%% ---------------------------------------------------------
% Remove invalid RR values
% ---------------------------------------------------------

valid = ...
    isfinite(rrWindow) & ...
    isfinite(rrTime);

rrValues = ...
    rrWindow(valid);

rrTimes = ...
    rrTime(valid);

if numel(rrValues) < 10
    return;
end

%% ---------------------------------------------------------
% Ensure unique timestamps
% ---------------------------------------------------------

[rrTimes,uniqueIdx] = ...
    unique(rrTimes);

rrValues = ...
    rrValues(uniqueIdx);

if numel(rrTimes) < 5
    return;
end

%% ---------------------------------------------------------
% Interpolate using actual timestamps
%
% No extrapolation is performed.
% ---------------------------------------------------------

interpTime = ...
    (rrTimes(1):1/Fs:rrTimes(end))';

if numel(interpTime) < 5
    return;
end

rrInterp = ...
    interp1( ...
    rrTimes, ...
    rrValues, ...
    interpTime, ...
    'linear');

if any(~isfinite(rrInterp))

    validInterp = ...
        isfinite(rrInterp);

    rrInterp = ...
        rrInterp(validInterp);

end

if numel(rrInterp) < 5
    return;
end

rrInterp = ...
    detrend(rrInterp);

%% ---------------------------------------------------------
% PSD
% ---------------------------------------------------------

[Pxx,freq] = ...
    simplePSD(rrInterp,Fs);

if isempty(Pxx)
    return;
end

Pxx = Pxx(:);
freq = freq(:);

%% ---------------------------------------------------------
% Frequency masks
% ---------------------------------------------------------

idxVLF = ...
    freq >= VLF_Low & ...
    freq < VLF_High;

idxLF = ...
    freq >= LF_Low & ...
    freq < LF_High;

idxHF = ...
    freq >= HF_Low & ...
    freq <= HF_High;

% Respiratory HRV band is explicitly clipped to the
% HRV frequency range so that it is a subset of the
% denominator used for RespiratoryPowerFraction_HRV.
respLower = ...
    max(RespLow,VLF_Low);

respUpper = ...
    min(RespHigh,HF_High);

idxResp = ...
    freq >= respLower & ...
    freq <= respUpper;

%% ---------------------------------------------------------
% Band powers
% ---------------------------------------------------------

VLFPower = ...
    safeTrapz(freq(idxVLF),Pxx(idxVLF));

LFPower = ...
    safeTrapz(freq(idxLF),Pxx(idxLF));

HFPower = ...
    safeTrapz(freq(idxHF),Pxx(idxHF));

if isfinite(VLFPower) && ...
        isfinite(LFPower) && ...
        isfinite(HFPower)

    TotalHRVPower = ...
        VLFPower+LFPower+HFPower;

end

%% ---------------------------------------------------------
% LF/HF ratio
% ---------------------------------------------------------

if isfinite(LFPower) && ...
        isfinite(HFPower) && ...
        HFPower > 0

    LF_HF_Ratio = ...
        LFPower/HFPower;

end

%% ---------------------------------------------------------
% Normalised LF/HF
% ---------------------------------------------------------

LFHFsum = ...
    LFPower+HFPower;

if isfinite(LFHFsum) && ...
        LFHFsum > 0

    LF_Normalised = ...
        LFPower/LFHFsum;

    HF_Normalised = ...
        HFPower/LFHFsum;

end

%% ---------------------------------------------------------
% HRV-band dominant frequency
% ---------------------------------------------------------

usefulIdx = ...
    freq >= VLF_Low & ...
    freq <= HF_High;

usefulPower = ...
    Pxx(usefulIdx);

usefulFreq = ...
    freq(usefulIdx);

validPower = ...
    isfinite(usefulPower) & ...
    usefulPower >= 0;

usefulPower = ...
    usefulPower(validPower);

usefulFreq = ...
    usefulFreq(validPower);

if ~isempty(usefulPower)

    [DominantHRVPower,idxMax] = ...
        max(usefulPower);

    DominantHRVFrequency = ...
        usefulFreq(idxMax);

end

%% ---------------------------------------------------------
% LF peak
% ---------------------------------------------------------

if any(idxLF)

    lfPower = ...
        Pxx(idxLF);

    lfFreq = ...
        freq(idxLF);

    validLF = ...
        isfinite(lfPower);

    lfPower = ...
        lfPower(validLF);

    lfFreq = ...
        lfFreq(validLF);

    if ~isempty(lfPower)

        [~,idx] = ...
            max(lfPower);

        LF_PeakFrequency = ...
            lfFreq(idx);

    end

end

%% ---------------------------------------------------------
% HF peak
% ---------------------------------------------------------

if any(idxHF)

    hfPower = ...
        Pxx(idxHF);

    hfFreq = ...
        freq(idxHF);

    validHF = ...
        isfinite(hfPower);

    hfPower = ...
        hfPower(validHF);

    hfFreq = ...
        hfFreq(validHF);

    if ~isempty(hfPower)

        [~,idx] = ...
            max(hfPower);

        HF_PeakFrequency = ...
            hfFreq(idx);

    end

end

%% ---------------------------------------------------------
% LF*HF interaction
% ---------------------------------------------------------

if isfinite(LFPower) && ...
        isfinite(HFPower)

    LF_HF_Power = ...
        LFPower*HFPower;

end

%% ---------------------------------------------------------
% HF power fraction
% ---------------------------------------------------------

if isfinite(HFPower) && ...
        isfinite(TotalHRVPower) && ...
        TotalHRVPower > 0

    HF_PowerFraction = ...
        HFPower/TotalHRVPower;

end

%% ---------------------------------------------------------
% HRV spectral features
% ---------------------------------------------------------

if ~isempty(usefulPower)

    HRVSpectralEntropy = ...
        spectralEntropy(usefulPower);

    totalSpecPower = ...
        sum(usefulPower);

    if totalSpecPower > 0

        HRVSpectralCentroid = ...
            sum(usefulFreq.*usefulPower)/ ...
            totalSpecPower;

        HRVSpectralBandwidth = ...
            sqrt(max(0, ...
            sum((usefulFreq- ...
            HRVSpectralCentroid).^2.* ...
            usefulPower)/ ...
            totalSpecPower));

    end

end

%% ---------------------------------------------------------
% Respiratory HRV power
% ---------------------------------------------------------

RespiratoryPower_HRV = ...
    safeTrapz( ...
    freq(idxResp), ...
    Pxx(idxResp));

if isfinite(RespiratoryPower_HRV) && ...
        isfinite(TotalHRVPower) && ...
        TotalHRVPower > 0

    RespiratoryPowerFraction_HRV = ...
        RespiratoryPower_HRV/TotalHRVPower;

    % Numerical protection: this is a fraction and must remain in [0,1].
    RespiratoryPowerFraction_HRV = ...
        max(0,min(1,RespiratoryPowerFraction_HRV));

end

end

%% ========================================================
% SAFE RATIO
% ========================================================

function value = safeRatio(numerator,denominator)

if ~isfinite(numerator) || ...
        ~isfinite(denominator) || ...
        abs(denominator) < eps

    value = NaN;

else

    value = ...
        numerator/denominator;

end

end

%% ========================================================
% COEFFICIENT OF VARIATION
% ========================================================

function value = coefficientVariation(x)

x = x(:);

x = x(isfinite(x));

if numel(x) < 2

    value = NaN;

    return;

end

mu = ...
    mean(x);

sigma = ...
    std(x,0);

if abs(mu) < eps

    value = NaN;

else

    value = ...
        sigma/abs(mu);

end

end

%% ========================================================
% SAFE TRAPZ
% ========================================================

function value = safeTrapz(x,y)

x = x(:);
y = y(:);

valid = ...
    isfinite(x) & ...
    isfinite(y);

x = x(valid);
y = y(valid);

if numel(x) < 2

    value = NaN;

    return;

end

value = ...
    trapz(x,y);

end

%% ========================================================
% SIMPLE PSD
% ========================================================
%
% Properly scaled one-sided periodogram.
%
% Pxx has units of signal^2/Hz.
%
% Integrating Pxx over frequency therefore gives signal
% variance approximately.
% ========================================================

function [Pxx,f] = simplePSD(x,Fs)

x = x(:);

x = x(isfinite(x));

if numel(x) < 4

    Pxx = [];
    f = [];

    return;

end

x = ...
    x-mean(x);

N = ...
    numel(x);

X = ...
    fft(x);

% Proper PSD scaling
P2 = ...
    abs(X).^2/(Fs*N);

nFreq = ...
    floor(N/2)+1;

Pxx = ...
    P2(1:nFreq);

% Correct one-sided scaling for both even and odd N
if mod(N,2) == 0

    if numel(Pxx) > 2

        Pxx(2:end-1) = ...
            2*Pxx(2:end-1);

    end

else

    if numel(Pxx) > 1

        Pxx(2:end) = ...
            2*Pxx(2:end);

    end

end

f = ...
    (Fs*(0:floor(N/2))/N)';

Pxx = ...
    Pxx(:);

f = ...
    f(:);

end

%% ========================================================
% SPECTRAL ENTROPY
% ========================================================

function value = spectralEntropy(P)

P = P(:);

P = ...
    P(isfinite(P) & P > 0);

if isempty(P)

    value = NaN;

    return;

end

P = ...
    P/sum(P);

value = ...
    -sum(P.*log2(P));

if numel(P) > 1

    value = ...
        value/log2(numel(P));

else

    value = 0;

end

% Numerical protection
value = ...
    max(0,min(1,value));

end

%% ========================================================
% FAST PERCENTILE
% ========================================================

function value = percentile_fast(x,p)

x = ...
    sort(x(:));

x = ...
    x(isfinite(x));

n = ...
    numel(x);

if n == 0

    value = NaN;

    return;

end

pos = ...
    1+(n-1)*p/100;

lower = ...
    floor(pos);

upper = ...
    ceil(pos);

if lower == upper

    value = ...
        x(lower);

else

    value = ...
        x(lower)+(pos-lower)* ...
        (x(upper)-x(lower));

end

end

%% ========================================================
% LONGEST CONTIGUOUS VALID RR SEGMENT
% ========================================================

function segment = longestContiguousSegment(x)

x = x(:);

valid = ...
    isfinite(x);

if ~any(valid)

    segment = [];

    return;

end

% Find starts and ends of valid runs
d = ...
    diff([false;valid;false]);

starts = ...
    find(d == 1);

ends = ...
    find(d == -1)-1;

lengths = ...
    ends-starts+1;

[~,idx] = ...
    max(lengths);

segment = ...
    x(starts(idx):ends(idx));

end

%% ========================================================
% LIMIT RR LENGTH FOR NONLINEAR FEATURES
% ========================================================

function x = limitRRLength(x,MaxN)

x = x(:);

if isempty(x)

    return;

end

if isinf(MaxN) || ...
        numel(x) <= MaxN

    return;

end

% Evenly sample across the available contiguous sequence
idx = ...
    round(linspace(1,numel(x),MaxN));

idx = ...
    unique(idx);

x = ...
    x(idx);

end

%% ========================================================
% FAST SAMPLE ENTROPY
% ========================================================

function value = sampleEntropyFast(x)

x = x(:);

x = ...
    x(isfinite(x));

N = ...
    numel(x);

if N < 20

    value = NaN;

    return;

end

m = 2;

r = ...
    0.2*std(x);

if r <= 0

    value = NaN;

    return;

end

Xm = ...
    zeros(N-m+1,m);

Xm1 = ...
    zeros(N-m,m+1);

for k = 1:m

    Xm(:,k) = ...
        x(k:k+N-m);

end

for k = 1:m+1

    Xm1(:,k) = ...
        x(k:k+N-m-1);

end

countM = 0;

for i = 1:size(Xm,1)-1

    d = ...
        max(abs(Xm(i+1:end,:)-Xm(i,:)),[],2);

    countM = ...
        countM+sum(d <= r);

end

countM1 = 0;

for i = 1:size(Xm1,1)-1

    d = ...
        max(abs(Xm1(i+1:end,:)-Xm1(i,:)),[],2);

    countM1 = ...
        countM1+sum(d <= r);

end

if countM <= 0 || ...
        countM1 <= 0

    value = NaN;

else

    value = ...
        -log(countM1/countM);

end

end

%% ========================================================
% FAST APPROXIMATE ENTROPY
% ========================================================

function value = approximateEntropyFast(x)

x = x(:);

x = ...
    x(isfinite(x));

N = ...
    numel(x);

if N < 20

    value = NaN;

    return;

end

m = 2;

r = ...
    0.2*std(x);

if r <= 0

    value = NaN;

    return;

end

phi = ...
    zeros(2,1);

for dimension = 1:2

    currentM = ...
        m+dimension-1;

    numPatterns = ...
        N-currentM+1;

    templates = ...
        zeros(numPatterns,currentM);

    for k = 1:currentM

        templates(:,k) = ...
            x(k:k+numPatterns-1);

    end

    C = ...
        zeros(numPatterns,1);

    for i = 1:numPatterns

        distances = ...
            max(abs(templates-templates(i,:)),[],2);

        C(i) = ...
            mean(distances <= r);

    end

    phi(dimension) = ...
        mean(log(C+eps));

end

value = ...
    phi(1)-phi(2);

end

%% ========================================================
% TURNING POINT RATIO
% ========================================================

function value = turningPointRatioFast(x)

x = x(:);

if numel(x) < 4

    value = NaN;

    return;

end

% Only use genuinely consecutive valid triples.
valid = ...
    isfinite(x);

if sum(valid) < 4

    value = NaN;

    return;

end

d1 = ...
    diff(x(1:end-1));

d2 = ...
    diff(x(2:end));

tripleValid = ...
    valid(1:end-2) & ...
    valid(2:end-1) & ...
    valid(3:end);

turningPoints = ...
    sum(d1(tripleValid).* ...
    d2(tripleValid) < 0);

numTriples = ...
    sum(tripleValid);

if numTriples <= 0

    value = NaN;

else

    value = ...
        turningPoints/numTriples;

end

end

%% ========================================================
% EDR-R FEATURES
% ========================================================

function features = edrFeaturesFast( ...
    t,signal,RespLow,RespHigh)

signal = signal(:);
t = t(:);

valid = ...
    isfinite(signal) & ...
    isfinite(t);

signal = ...
    signal(valid);

t = ...
    t(valid);

if numel(signal) < 10

    features = ...
        nan(1,10);

    return;

end

[ t,uniqueIdx ] = ...
    unique(t);

signal = ...
    signal(uniqueIdx);

if numel(signal) < 10

    features = ...
        nan(1,10);

    return;

end

MeanValue = ...
    mean(signal);

SDValue = ...
    std(signal,0);

RangeValue = ...
    max(signal)-min(signal);

duration = ...
    t(end)-t(1);

if duration <= 0

    features = ...
        nan(1,10);

    return;

end

EDRFs = 4;

uniformTime = ...
    (t(1):1/EDRFs:t(end))';

if numel(uniformTime) < 5

    features = ...
        nan(1,10);

    return;

end

uniformSignal = ...
    interp1( ...
    t, ...
    signal, ...
    uniformTime, ...
    'linear');

validInterp = ...
    isfinite(uniformSignal);

uniformSignal = ...
    uniformSignal(validInterp);

if numel(uniformSignal) < 5

    features = ...
        nan(1,10);

    return;

end

uniformSignal = ...
    detrend(uniformSignal);

[Pxx,f] = ...
    simplePSD(uniformSignal,EDRFs);

if isempty(Pxx)

    features = ...
        nan(1,10);

    return;

end

Pxx = Pxx(:);
f = f(:);

idxResp = ...
    f >= RespLow & ...
    f <= RespHigh;

RespiratoryPower = ...
    safeTrapz( ...
    f(idxResp), ...
    Pxx(idxResp));

if any(idxResp)

    respPower = ...
        Pxx(idxResp);

    respFreq = ...
        f(idxResp);

    validResp = ...
        isfinite(respPower) & ...
        isfinite(respFreq) & ...
        respPower >= 0;

    respPower = ...
        respPower(validResp);

    respFreq = ...
        respFreq(validResp);

    if ~isempty(respPower)

        [DominantPower,idx] = ...
            max(respPower);

        DominantFrequency = ...
            respFreq(idx);

        RespiratoryRate = ...
            DominantFrequency*60;

        SpectralEntropy = ...
            spectralEntropy(respPower);

    else

        DominantPower = NaN;
        DominantFrequency = NaN;
        RespiratoryRate = NaN;
        SpectralEntropy = NaN;

    end

else

    DominantPower = NaN;
    DominantFrequency = NaN;
    RespiratoryRate = NaN;
    SpectralEntropy = NaN;

end

totalPower = ...
    safeTrapz(f,Pxx);

if isfinite(RespiratoryPower) && ...
        isfinite(totalPower) && ...
        totalPower > 0

    RelativeRespPower = ...
        RespiratoryPower/totalPower;

else

    RelativeRespPower = NaN;

end

centered = ...
    uniformSignal-mean(uniformSignal);

ZeroCrossingRate = ...
    sum(centered(1:end-1).* ...
    centered(2:end) < 0)/duration;

features = [ ...
    MeanValue
    SDValue
    RangeValue
    RespiratoryPower
    DominantFrequency
    RespiratoryRate
    DominantPower
    SpectralEntropy
    RelativeRespPower
    ZeroCrossingRate]';

end

%% ========================================================
% BASIC EDR FEATURES
% ========================================================

function features = edrFeaturesBasicFast( ...
    t,signal,RespLow,RespHigh)

signal = signal(:);
t = t(:);

valid = ...
    isfinite(signal) & ...
    isfinite(t);

signal = ...
    signal(valid);

t = ...
    t(valid);

if numel(signal) < 10

    features = ...
        nan(1,6);

    return;

end

[t,uniqueIdx] = ...
    unique(t);

signal = ...
    signal(uniqueIdx);

if numel(signal) < 10

    features = ...
        nan(1,6);

    return;

end

duration = ...
    t(end)-t(1);

if duration <= 0

    features = ...
        nan(1,6);

    return;

end

EDRFs = 4;

uniformTime = ...
    (t(1):1/EDRFs:t(end))';

if numel(uniformTime) < 5

    features = ...
        nan(1,6);

    return;

end

uniformSignal = ...
    interp1( ...
    t, ...
    signal, ...
    uniformTime, ...
    'linear');

validInterp = ...
    isfinite(uniformSignal);

uniformSignal = ...
    uniformSignal(validInterp);

if numel(uniformSignal) < 5

    features = ...
        nan(1,6);

    return;

end

uniformSignal = ...
    detrend(uniformSignal);

[Pxx,f] = ...
    simplePSD(uniformSignal,EDRFs);

if isempty(Pxx)

    features = ...
        nan(1,6);

    return;

end

Pxx = Pxx(:);
f = f(:);

idxResp = ...
    f >= RespLow & ...
    f <= RespHigh;

RespiratoryPower = ...
    safeTrapz( ...
    f(idxResp), ...
    Pxx(idxResp));

if any(idxResp)

    respPower = ...
        Pxx(idxResp);

    respFreqs = ...
        f(idxResp);

    validResp = ...
        isfinite(respPower) & ...
        isfinite(respFreqs) & ...
        respPower >= 0;

    respPower = ...
        respPower(validResp);

    respFreqs = ...
        respFreqs(validResp);

    if ~isempty(respPower)

        [DominantPower,idx] = ...
            max(respPower);

        DominantFrequency = ...
            respFreqs(idx);

        RespiratoryRate = ...
            DominantFrequency*60;

        SpectralEntropy = ...
            spectralEntropy(respPower);

    else

        DominantPower = NaN;
        DominantFrequency = NaN;
        RespiratoryRate = NaN;
        SpectralEntropy = NaN;

    end

else

    DominantPower = NaN;
    DominantFrequency = NaN;
    RespiratoryRate = NaN;
    SpectralEntropy = NaN;

end

totalPower = ...
    safeTrapz(f,Pxx);

if isfinite(RespiratoryPower) && ...
        isfinite(totalPower) && ...
        totalPower > 0

    RelativeRespPower = ...
        RespiratoryPower/totalPower;

else

    RelativeRespPower = NaN;

end

features = [ ...
    RespiratoryPower
    DominantFrequency
    RespiratoryRate
    DominantPower
    SpectralEntropy
    RelativeRespPower]';

end

%% ========================================================
% RSA FEATURES
% ========================================================

function [RespPower, ...
    DominantFrequency, ...
    RespiratoryRate, ...
    DominantPower, ...
    SpectralEntropy] = ...
    rsaFeaturesFast( ...
    rrWindow, ...
    rrTime, ...
    Fs, ...
    RespLow, ...
    RespHigh)

RespPower = NaN;
DominantFrequency = NaN;
RespiratoryRate = NaN;
DominantPower = NaN;
SpectralEntropy = NaN;

rrWindow = rrWindow(:);
rrTime = rrTime(:);

valid = ...
    isfinite(rrWindow) & ...
    isfinite(rrTime);

rrValues = ...
    rrWindow(valid);

rrTimes = ...
    rrTime(valid);

if numel(rrValues) < 10

    return;

end

[rrTimes,idx] = ...
    unique(rrTimes);

rrValues = ...
    rrValues(idx);

if numel(rrTimes) < 5

    return;

end

interpTime = ...
    (rrTimes(1):1/Fs:rrTimes(end))';

if numel(interpTime) < 5

    return;

end

rrInterp = ...
    interp1( ...
    rrTimes, ...
    rrValues, ...
    interpTime, ...
    'linear');

validInterp = ...
    isfinite(rrInterp);

rrInterp = ...
    rrInterp(validInterp);

if numel(rrInterp) < 5

    return;

end

rrInterp = ...
    detrend(rrInterp);

[Pxx,f] = ...
    simplePSD(rrInterp,Fs);

if isempty(Pxx)

    return;

end

Pxx = Pxx(:);
f = f(:);

idxResp = ...
    f >= RespLow & ...
    f <= RespHigh;

RespPower = ...
    safeTrapz( ...
    f(idxResp), ...
    Pxx(idxResp));

if any(idxResp)

    respPower = ...
        Pxx(idxResp);

    respFreqs = ...
        f(idxResp);

    validResp = ...
        isfinite(respPower) & ...
        isfinite(respFreqs) & ...
        respPower >= 0;

    respPower = ...
        respPower(validResp);

    respFreqs = ...
        respFreqs(validResp);

    if ~isempty(respPower)

        [DominantPower,idxMax] = ...
            max(respPower);

        DominantFrequency = ...
            respFreqs(idxMax);

        RespiratoryRate = ...
            DominantFrequency*60;

        SpectralEntropy = ...
            spectralEntropy(respPower);

    end

end

end

%% ========================================================
% KURTOSIS
% ========================================================

function value = kurtosisManualFast(x)

x = x(:);

x = ...
    x(isfinite(x));

if numel(x) < 4

    value = NaN;

    return;

end

mu = ...
    mean(x);

sigma = ...
    std(x,0);

if sigma <= 0

    value = 0;

    return;

end

value = ...
    mean(((x-mu)/sigma).^4);

end