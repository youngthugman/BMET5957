function qrs = alignQRS(qrs, signal, searchSamples)
% hello
%    ALIGNQRS Refine each QRS to a consistent nearby fiducial point.
%   Prefer a strong positive R peak if present. Otherwise keep the strongest
%   absolute deflection, which protects inverted/negative QRS cases.

    qrs = qrs(:);
    baselineSamples = 25;       % 250 ms each side at 100 Hz
    positiveRatio = 0.35;       % positive peak must be at least 35% of abs peak

    for k = 1:length(qrs)
        idx1 = max(1, qrs(k) - searchSamples);
        idx2 = min(length(signal), qrs(k) + searchSamples);

        base1 = max(1, qrs(k) - baselineSamples);
        base2 = min(length(signal), qrs(k) + baselineSamples);

        localBaseline = median(signal(base1:base2), 'omitnan');
        localSegment = signal(idx1:idx2) - localBaseline;

        [absVal, absIdx] = max(abs(localSegment));
        [posVal, posIdx] = max(localSegment);

        if posVal > positiveRatio * absVal
            localIdx = posIdx;          % use positive R peak
        else
            localIdx = absIdx;          % fallback for inverted/negative QRS
        end

        qrs(k) = idx1 + localIdx - 1;
    end

    qrs = unique(qrs, 'stable');
end
