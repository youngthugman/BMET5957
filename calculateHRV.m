function [avgRR, sdRR, RMSSD, pNN50, LF, HF, LF_HFratio] = calculateHRV(qrs, signalLength, fs)
%CALCULATEHRV Calculate HRV parameters from cleaned QRS detections.
% hello
% Internal units:
%   avgRR, sdRR, RMSSD = seconds
%   LF, HF = seconds^2
%   pNN50 = percent
%   LF_HFratio = unitless
    avgRR = NaN;
    sdRR = NaN;
    RMSSD = NaN;
    pNN50 = NaN;
    LF = NaN;
    HF = NaN;
    LF_HFratio = NaN;
    qrs = qrs(:);
    if length(qrs) < 2
        return
    end
    % ---------------------------------------------------------------------
    % RR intervals and non-normal interval marking
    % ---------------------------------------------------------------------
    RR = diff(qrs) / fs;                 % RR intervals in seconds
    qrsRR = qrs(2:end);                  % sample linked to each RR interval
    normalRR = RR(RR > 0.30 & RR < 2.00);
    if isempty(normalRR)
        return
    end
    typicalRR = median(normalRR, 'omitnan');
    % Replace the badRR definition:
    badRR = RR < 0.30 | RR > 2.00;   % physiological hard limits
    
    % Then iterative local outlier removal (more robust than global typicalRR)
    for iter = 1:5
        validForMedian = RR(~badRR);
        if isempty(validForMedian), break; end
        localMed = medfilt1(RR, 9, 'omitnan');   % 5-beat local median
        badRR = badRR | abs(RR - localMed) > 0.15 * localMed;  % 20% deviation
    end
    RR(badRR) = NaN;                     % keep gaps as missing values
    
    % Remove RR intervals where the successive difference is physiologically
    % implausible — catches ectopic pairs that slip past the local median filter
    RRdiff_all = abs(diff(RR));
    suspectPairs = RRdiff_all > 0.20;   % 200ms jump between consecutive beats
    suspectIdx = find(suspectPairs);
    for k = 1:length(suspectIdx)
        % Mark both beats in the suspect pair as bad
        RR(suspectIdx(k)) = NaN;
        RR(suspectIdx(k) + 1) = NaN;
    end
    % ---------------------------------------------------------------------
    % Time-domain HRV in 5-minute windows, then average windows
    % ---------------------------------------------------------------------
    epochSamples = 5 * 60 * fs;
    nEpochs = floor(signalLength / epochSamples);
    avgEpoch = nan(nEpochs, 1);
    sdEpoch = nan(nEpochs, 1);
    rmssdEpoch = nan(nEpochs, 1);
    pnn50Epoch = nan(nEpochs, 1);
    for e = 1:nEpochs
            idx1 = (e - 1) * epochSamples + 1;
            idx2 = e * epochSamples;
            RRseg = RR(qrsRR >= idx1 & qrsRR <= idx2);
            validRR = RRseg(~isnan(RRseg));
    if length(validRR) >= 2
                avgEpoch(e) = mean(validRR, 'omitnan');
                sdEpoch(e) = std(validRR, 'omitnan');
    end
            RRdiff = diff(RRseg);            % BUG: diffing RRseg which still has NaNs
            validDiff = RRdiff(~isnan(RRdiff));
    if ~isempty(validDiff) && ~isempty(validRR)
                rmssdEpoch(e) = sqrt(mean(validDiff .^ 2, 'omitnan'));
                pnn50Epoch(e) = sum(abs(validDiff) > 0.05) / length(validRR) * 100;
    end
    end
    avgRR = mean(avgEpoch, 'omitnan');
    sdRR = mean(sdEpoch, 'omitnan');
    RMSSD = mean(rmssdEpoch, 'omitnan');
    pNN50 = mean(pnn50Epoch, 'omitnan');
    % ---------------------------------------------------------------------
    % Frequency-domain HRV using Lomb-Scargle PSD, 1-minute segments
    % ---------------------------------------------------------------------
    segmentSamples = 60 * fs;
    nSegments = floor(signalLength / segmentSamples);
    freq = 0.5 * (0:99) / 100;           % 100 bins from 0 to 0.495 Hz
    PSD = nan(nSegments, length(freq));
    tRR = qrsRR / fs;                    % RR timing in seconds
    for s = 1:nSegments
        idx1 = (s - 1) * segmentSamples + 1;
        idx2 = s * segmentSamples;
        inSeg = qrsRR >= idx1 & qrsRR <= idx2;
        RRseg = RR(inSeg);
        tSeg = tRR(inSeg);
        good = ~isnan(RRseg) & ~isnan(tSeg);
        RRseg = RRseg(good);
        tSeg = tSeg(good);
        if length(RRseg) < 4
            continue
        end
        if length(RRseg) < 6
            continue
        end
        
        totalDuration = tSeg(end) - tSeg(1);
        if totalDuration < 20
            continue
        end
        [tSeg, uniqueIdx] = unique(tSeg, 'stable');
        RRseg = RRseg(uniqueIdx);
        if length(RRseg) < 4 || sum(RRseg, 'omitnan') <= 30
            continue
        end
        RRseg = RRseg - mean(RRseg, 'omitnan');
        PSD(s, :) = plomb(RRseg, tSeg, 'psd', freq);
    end
    PSDavg = mean(PSD, 1, 'omitnan');
    LFidx = freq >= 0.04 & freq < 0.15;
    HFidx = freq >= 0.15 & freq <= 0.40;
    LF = trapz(freq(LFidx), PSDavg(LFidx));
    HF = trapz(freq(HFidx), PSDavg(HFidx));
    if HF > 0
        LF_HFratio = LF / HF;
    end
end
