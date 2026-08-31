function qrs = removeCloseQRS(qrs, signal, minDistance)
% hello
% REMOVECLOSEQRS Remove duplicate QRS detections closer than minDistance.
%   If two detections are too close, keep the one with larger absolute ECG value.

    qrs = qrs(:);

    if length(qrs) < 2
        return
    end

    q = 2;

    while q <= length(qrs)
        if qrs(q) - qrs(q - 1) < minDistance

            ampPrev = abs(signal(qrs(q - 1)));
            ampCurr = abs(signal(qrs(q)));

            if ampCurr > ampPrev
                qrs(q - 1) = [];
            else
                qrs(q) = [];
            end

        else
            q = q + 1;
        end
    end
end
