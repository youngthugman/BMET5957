function [signalFiltered, badSegment] = simpleFilters(signal, fs)
% hello
% SIMPLEFILTERS Offline ECG preprocessing.
%   1. Mark bad signal regions using whole-record local range.
%   2. Clip large amplitude spikes before Pan-Tompkins.
    signal = signal(:);
    % ---------------------------------------------------------------------
    % Offline signal-quality check
    % ---------------------------------------------------------------------
    halfWindow = round(2.5 * fs);    % centred 5-second activity window
    rawLocalRange = movmax(signal, [halfWindow halfWindow]) - ...
                    movmin(signal, [halfWindow halfWindow]);
    typicalRange = median(rawLocalRange, 'omitnan');
    tooFlat = rawLocalRange < max(50, 0.20 * typicalRange);
    tooWild = rawLocalRange > 3.5 * typicalRange; % 4 had lowest mape, 3 has highest F1.
    badSegment = tooFlat | tooWild;
    % ---------------------------------------------------------------------
    % Amplitude clipping from whole-record epoch distribution
    % ---------------------------------------------------------------------
    epochSamples = round(0.8 * fs);
    nEpochs = ceil(length(signal) / epochSamples);
    epochMax = nan(nEpochs, 1);
    for e = 1:nEpochs
        idx1 = (e - 1) * epochSamples + 1;
        idx2 = min(e * epochSamples, length(signal));
        epochMax(e) = max(abs(signal(idx1:idx2)));
    end
    clipThreshold = mean(epochMax, 'omitnan') + 1.2 * std(epochMax, 'omitnan');
    signalFiltered = signal;
    signalFiltered(signalFiltered > clipThreshold) = clipThreshold;
    signalFiltered(signalFiltered < -clipThreshold) = -clipThreshold;
end
