function [TP, FP, FN] = qrsMetrics(qrs, qrsRef, tol)
% hello
% QRSMETRICS Count TP, FP, FN using tolerance in samples.

    qrs = qrs(:);
    qrsRef = qrsRef(:);

    matched = false(length(qrs), 1);

    TP = 0;
    FN = 0;

    for j = 1:length(qrsRef)
        diffs = abs(qrs - qrsRef(j));
        candidates = find(diffs <= tol & ~matched);

        if isempty(candidates)
            FN = FN + 1;
        else
            [~, idx] = min(diffs(candidates));
            matched(candidates(idx)) = true;
            TP = TP + 1;
        end
    end

    FP = sum(~matched);
end
