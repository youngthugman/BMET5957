function [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ecg,fs,gr)
% hello
%% function [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ecg,fs,gr)
% Pan-Tompkins QRS detector with:
%   1. Sliding-window median peak level estimator
%   2. Parabolic interpolation for sub-sample fiducial mark accuracy

%
%% Inputs
%   ecg  : raw ECG vector (1D)
%   fs   : sampling frequency in Hz 
%   gr   : plot flag — 1 to plot, 0 to suppress
%
%% Outputs
%   qrs_amp_raw : amplitudes of detected R waves (bandpass signal)
%   qrs_i_raw   : sample indices of detected R waves (fiducial marks)
%                 NOTE: add delay before comparing to expert annotations
%                 in absolute time:  qrs_i_corrected = qrs_i_raw + delay
%   delay       : total filter delay in samples

if ~isvector(ecg)
    error('ecg must be a row or column vector');
end
if nargin < 3
    gr = 1;
end
ecg = ecg(:);

%% ======================= Initialize =============================== %%
delay         = 0;
skip          = 0;
m_selected_RR = 0;
mean_RR       = 0;
ser_back      = 0;
ax            = zeros(1,6);

%% ======================= Tunable parameters ====================== %%
N_QRS      = 13;    % median window for QRS peaks 
N_NOISE    = 6;     % median window for noise peaks
TC         = 0.222; % threshold coefficient: DT = NPL + TC*|QRSPL-NPL|
SB_FRAC    = 0.5;   % search-back acceptance as fraction of THR_SIG
                    
SB_TRIGGER = 1.50;  % search-back fires after this multiple of mean RR
                    
RR_WIN     = 10;    % number of regular beats used for RR mean
RR_LO      = 0.88;  % lower acceptance bound for regular beats (fraction)
RR_HI      = 1.20;  % upper acceptance bound for regular beats (fraction)

%% ======================= Median buffers ========================== %%
qrs_peak_buf    = zeros(1, N_QRS);
qrs_peak_buf1   = zeros(1, N_QRS);
noise_peak_buf  = zeros(1, N_NOISE);
noise_peak_buf1 = zeros(1, N_NOISE);
qrs_buf_n       = 0;
noise_buf_n     = 0;

%% ======================= RR interval buffer ====================== %%
rr_reg_buf = zeros(1, RR_WIN);
rr_reg_n   = 0;

%% ============ Noise cancellation — filtering (5-15 Hz) =========== %%
if fs == 200
    ecg = ecg - mean(ecg);

    Wn = 12*2/fs; N = 3;
    [a,b] = butter(N, Wn, 'low');
    ecg_l = filtfilt(a, b, ecg);
    ecg_l = ecg_l / max(abs(ecg_l));

    if gr
        figure;
        ax(1) = subplot(321); plot(ecg);   axis tight; title('Raw signal');
        ax(2) = subplot(322); plot(ecg_l); axis tight; title('Low pass filtered');
    end

    Wn = 5*2/fs; N = 3;
    [a,b] = butter(N, Wn, 'high');
    ecg_h = filtfilt(a, b, ecg_l);
    ecg_h = ecg_h / max(abs(ecg_h));

    if gr
        ax(3) = subplot(323); plot(ecg_h); axis tight; title('High pass filtered');
    end
else
    f1 = 6; f2 = 16;
    Wn = [f1 f2]*2/fs; N = 3;
    [a,b] = butter(N, Wn);
    ecg_h = filtfilt(a, b, ecg);
    ecg_h = ecg_h / max(abs(ecg_h));

    if gr
        ax(1) = subplot(3,2,[1 2]); plot(ecg);   axis tight; title('Raw Signal');
        ax(3) = subplot(323);       plot(ecg_h); axis tight; title('Band pass filtered');
    end
end

%% ==================== Derivative filter ========================== %%
if fs ~= 200
    int_c = (5-1)/(fs*1/40);
    b = interp1(1:5, [1 2 0 -2 -1].*(1/8)*fs, 1:int_c:5);
else
    b = [1 2 0 -2 -1].*(1/8)*fs;
end
ecg_d = filtfilt(b, 1, ecg_h);
ecg_d = ecg_d / max(ecg_d);

if gr
    ax(4) = subplot(324); plot(ecg_d); axis tight; title('Derivative filtered');
end

%% ==================== Squaring ==================================== %%
ecg_s = ecg_d.^2;

if gr
    ax(5) = subplot(325); plot(ecg_s); axis tight; title('Squared');
end

%% ==================== Moving average (150 ms) ==================== %%
ecg_m = conv(ecg_s, ones(1, round(0.150*fs)) / round(0.150*fs));
delay = delay + round(0.150*fs)/2;

if gr
    ax(6) = subplot(326); plot(ecg_m); axis tight;
    title('Moving average — black:noise, green:threshold, red:signal, magenta:QRS');
end

%% ===================== Peak detection ============================ %%
[pks, locs] = findpeaks(ecg_m, 'MINPEAKDISTANCE', round(0.2*fs));

%% =================== Allocate output buffers ==================== %%
LLp         = length(pks);
qrs_c       = zeros(1, LLp);
qrs_i       = zeros(1, LLp);
qrs_i_raw   = zeros(1, LLp);
qrs_amp_raw = zeros(1, LLp);
nois_c      = zeros(1, LLp);
nois_i      = zeros(1, LLp);
SIGL_buf    = zeros(1, LLp);
NOISL_buf   = zeros(1, LLp);
THRS_buf    = zeros(1, LLp);
SIGL_buf1   = zeros(1, LLp);
NOISL_buf1  = zeros(1, LLp);
THRS_buf1   = zeros(1, LLp);

%% ========= Seed estimators from 2-second training window ========= %%
SIG_LEV    = max(ecg_m(1:2*fs)) * 1/3;
NOISE_LEV  = mean(ecg_m(1:2*fs)) * 1/2;
THR_SIG    = SIG_LEV;
THR_NOISE  = NOISE_LEV;

SIG_LEV1   = max(ecg_h(1:2*fs)) * 1/3;
NOISE_LEV1 = mean(ecg_h(1:2*fs)) * 1/2;
THR_SIG1   = SIG_LEV1;
THR_NOISE1 = NOISE_LEV1;

%% =================== Decision rule loop ========================== %%
Beat_C      = 0;
Beat_C1     = 0;
Noise_Count = 0;
win_start   = 1;   % initialise so it is always defined

for i = 1:LLp

    %% --- Locate bandpass peak with parabolic sub-sample refinement --- %%
    win_start = locs(i) - round(0.150*fs);

    if win_start >= 1 && locs(i) <= length(ecg_h)
        seg       = ecg_h(win_start : locs(i));
        [y_i, xi] = max(seg);
        x_i       = parabolic_peak(seg, xi);

    elseif i == 1
        win_start  = 1;
        seg        = ecg_h(1 : locs(i));
        [y_i, xi]  = max(seg);
        x_i        = parabolic_peak(seg, xi);
        ser_back   = 1;

    else  % locs(i) >= length(ecg_h)
        win_start  = locs(i) - round(0.150*fs);
        seg        = ecg_h(win_start : end);
        [y_i, xi]  = max(seg);
        x_i        = parabolic_peak(seg, xi);
    end

    %% --- RR interval update from fiducial marks ------------------- %%
    if Beat_C1 >= 2
        comp = qrs_i_raw(Beat_C1) - qrs_i_raw(Beat_C1 - 1);

        if rr_reg_n == 0
            rr_reg_n  = rr_reg_n + 1;
            rr_reg_buf(mod(rr_reg_n-1, RR_WIN)+1) = comp;
            m_selected_RR = comp;
            mean_RR       = comp;
        else
            ref_RR = mean(rr_reg_buf(1:min(rr_reg_n, RR_WIN)));
            if comp >= RR_LO*ref_RR && comp <= RR_HI*ref_RR
                rr_reg_n = rr_reg_n + 1;
                rr_reg_buf(mod(rr_reg_n-1, RR_WIN)+1) = comp;
                m_selected_RR = mean(rr_reg_buf(1:min(rr_reg_n, RR_WIN)));
            end
           
            mean_RR = mean(rr_reg_buf(1:min(rr_reg_n, RR_WIN)));
        end
    end

    if m_selected_RR
        test_m = m_selected_RR;
    elseif mean_RR
        test_m = mean_RR;
    else
        test_m = 0;
    end

    %% --- Search back -------------------------------------------- %%
    if test_m
        if (locs(i) - qrs_i(Beat_C)) >= round(SB_TRIGGER * test_m)

            sb_start = qrs_i(Beat_C) + round(0.200*fs);
            sb_end   = locs(i)       - round(0.200*fs);

            if sb_start < sb_end   % guard against invalid range
                [pks_temp, locs_temp] = max(ecg_m(sb_start : sb_end));
                locs_temp = sb_start + locs_temp - 1;

              
                if pks_temp > SB_FRAC * THR_SIG

                    Beat_C        = Beat_C + 1;
                    qrs_c(Beat_C) = pks_temp;
                    qrs_i(Beat_C) = locs_temp;

                    % Median update — search-back QRS (MVI signal)
                    qrs_buf_n = qrs_buf_n + 1;
                    qrs_peak_buf(mod(qrs_buf_n-1, N_QRS)+1) = pks_temp;
                    n_q       = min(qrs_buf_n, N_QRS);
                    SIG_LEV   = median(qrs_peak_buf(1:n_q));

                    % Fiducial mark — parabolic refinement in bandpass sig
                    sb_ws = locs_temp - round(0.150*fs);
                    if sb_ws >= 1 && locs_temp <= length(ecg_h)
                        seg_t         = ecg_h(sb_ws : locs_temp);
                        [y_i_t, xi_t] = max(seg_t);
                        xi_t_ref      = parabolic_peak(seg_t, xi_t);
                    else
                        sb_ws         = max(1, sb_ws);
                        seg_t         = ecg_h(sb_ws : min(length(ecg_h), locs_temp));
                        [y_i_t, xi_t] = max(seg_t);
                        xi_t_ref      = parabolic_peak(seg_t, xi_t);
                    end

                    if y_i_t > THR_NOISE1
                        Beat_C1              = Beat_C1 + 1;
                        qrs_i_raw(Beat_C1)   = round(sb_ws + xi_t_ref - 1);
                        qrs_amp_raw(Beat_C1) = y_i_t;

                        % Median update — search-back QRS (bandpass)
                        qrs_peak_buf1(mod(qrs_buf_n-1, N_QRS)+1) = y_i_t;
                        n_q1     = min(qrs_buf_n, N_QRS);
                        SIG_LEV1 = median(qrs_peak_buf1(1:n_q1));
                    end

                    not_nois   = 1;
                    THR_SIG    = NOISE_LEV  + TC * abs(SIG_LEV  - NOISE_LEV);
                    THR_NOISE  = 0.5 * THR_SIG;
                    THR_SIG1   = NOISE_LEV1 + TC * abs(SIG_LEV1 - NOISE_LEV1);
                    THR_NOISE1 = 0.5 * THR_SIG1;
                end
            end
        else
            not_nois = 0;
        end
    end

    %% ============ QRS / noise classification ==================== %%
    if pks(i) >= THR_SIG

        % T-wave rejection — within 360 ms of last QRS
        if Beat_C >= 3
            if (locs(i) - qrs_i(Beat_C)) <= round(0.360*fs)
                Slope1 = mean(diff(ecg_m(locs(i)       - round(0.075*fs) : locs(i))));
                Slope2 = mean(diff(ecg_m(qrs_i(Beat_C) - round(0.075*fs) : qrs_i(Beat_C))));

                if abs(Slope1) <= abs(0.5 * Slope2)
                    Noise_Count = Noise_Count + 1;
                    nois_c(Noise_Count) = pks(i);
                    nois_i(Noise_Count) = locs(i);
                    skip = 1;

                    noise_buf_n = noise_buf_n + 1;
                    noise_peak_buf(mod(noise_buf_n-1,  N_NOISE)+1) = pks(i);
                    noise_peak_buf1(mod(noise_buf_n-1, N_NOISE)+1) = y_i;
                    n_n        = min(noise_buf_n, N_NOISE);
                    NOISE_LEV  = median(noise_peak_buf(1:n_n));
                    NOISE_LEV1 = median(noise_peak_buf1(1:n_n));
                else
                    skip = 0;
                end
            end
        end

        if skip == 0
            % Confirmed QRS
            Beat_C        = Beat_C + 1;
            qrs_c(Beat_C) = pks(i);
            qrs_i(Beat_C) = locs(i);

            % Median update — confirmed QRS (MVI signal)
            qrs_buf_n = qrs_buf_n + 1;
            qrs_peak_buf(mod(qrs_buf_n-1, N_QRS)+1) = pks(i);
            n_q       = min(qrs_buf_n, N_QRS);
            SIG_LEV   = median(qrs_peak_buf(1:n_q));

            if y_i >= THR_SIG1
                Beat_C1 = Beat_C1 + 1;
                if ser_back
                    qrs_i_raw(Beat_C1) = round(x_i);
                else
                    qrs_i_raw(Beat_C1) = round(win_start + x_i - 1);
                end
                qrs_amp_raw(Beat_C1) = y_i;

                % Median update — confirmed QRS (bandpass signal)
                qrs_peak_buf1(mod(qrs_buf_n-1, N_QRS)+1) = y_i;
                n_q1     = min(qrs_buf_n, N_QRS);
                SIG_LEV1 = median(qrs_peak_buf1(1:n_q1));
            end
        end

    elseif (THR_NOISE <= pks(i)) && (pks(i) < THR_SIG)
        noise_buf_n = noise_buf_n + 1;
        noise_peak_buf(mod(noise_buf_n-1,  N_NOISE)+1) = pks(i);
        noise_peak_buf1(mod(noise_buf_n-1, N_NOISE)+1) = y_i;
        n_n        = min(noise_buf_n, N_NOISE);
        NOISE_LEV  = median(noise_peak_buf(1:n_n));
        NOISE_LEV1 = median(noise_peak_buf1(1:n_n));

    elseif pks(i) < THR_NOISE
        Noise_Count = Noise_Count + 1;
        nois_c(Noise_Count) = pks(i);
        nois_i(Noise_Count) = locs(i);

        noise_buf_n = noise_buf_n + 1;
        noise_peak_buf(mod(noise_buf_n-1,  N_NOISE)+1) = pks(i);
        noise_peak_buf1(mod(noise_buf_n-1, N_NOISE)+1) = y_i;
        n_n        = min(noise_buf_n, N_NOISE);
        NOISE_LEV  = median(noise_peak_buf(1:n_n));
        NOISE_LEV1 = median(noise_peak_buf1(1:n_n));
    end

    %% ============ Threshold update (Hamilton & Tompkins eq. 1) == %%
    THR_SIG    = NOISE_LEV  + TC * abs(SIG_LEV  - NOISE_LEV);
    THR_NOISE  = 0.5 * THR_SIG;
    THR_SIG1   = NOISE_LEV1 + TC * abs(SIG_LEV1 - NOISE_LEV1);
    THR_NOISE1 = 0.5 * THR_SIG1;

    %% Track buffers for plotting ---------------------------------- %%
    SIGL_buf(i)   = SIG_LEV;
    NOISL_buf(i)  = NOISE_LEV;
    THRS_buf(i)   = THR_SIG;
    SIGL_buf1(i)  = SIG_LEV1;
    NOISL_buf1(i) = NOISE_LEV1;
    THRS_buf1(i)  = THR_SIG1;

    %% Reset per-iteration flags ---------------------------------- %%
    skip     = 0;
    not_nois = 0;
    ser_back = 0;
end

%% ===================== Trim output arrays ======================== %%
qrs_i_raw   = qrs_i_raw(1:Beat_C1);
qrs_amp_raw = qrs_amp_raw(1:Beat_C1);
qrs_c       = qrs_c(1:Beat_C);
qrs_i       = qrs_i(1:Beat_C);

%% ===================== Plotting ================================== %%
if gr
    hold on; scatter(qrs_i, qrs_c, 'm');
    hold on; plot(locs, NOISL_buf, '--k', 'LineWidth', 2);
    hold on; plot(locs, SIGL_buf,  '--r', 'LineWidth', 2);
    hold on; plot(locs, THRS_buf,  '--g', 'LineWidth', 2);
    if any(ax)
        ax(~ax) = [];
        linkaxes(ax, 'x');
        zoom on;
    end

    figure;
    az(1) = subplot(311);
    plot(ecg_h); title('QRS on filtered signal'); axis tight;
    hold on; scatter(qrs_i_raw, qrs_amp_raw, 'm');
    hold on; plot(locs, NOISL_buf1, 'LineWidth', 2, 'Linestyle', '--', 'color', 'k');
    hold on; plot(locs, SIGL_buf1,  'LineWidth', 2, 'Linestyle', '-.', 'color', 'r');
    hold on; plot(locs, THRS_buf1,  'LineWidth', 2, 'Linestyle', '-.', 'color', 'g');

    az(2) = subplot(312); plot(ecg_m);
    title('MVI signal — noise (black), signal (red), threshold (green)'); axis tight;
    hold on; scatter(qrs_i, qrs_c, 'm');
    hold on; plot(locs, NOISL_buf, 'LineWidth', 2, 'Linestyle', '--', 'color', 'k');
    hold on; plot(locs, SIGL_buf,  'LineWidth', 2, 'Linestyle', '-.', 'color', 'r');
    hold on; plot(locs, THRS_buf,  'LineWidth', 2, 'Linestyle', '-.', 'color', 'g');

    az(3) = subplot(313);
    plot(ecg - mean(ecg)); title('Detected QRS on raw ECG'); axis tight;
    line(repmat(qrs_i_raw, [2 1]), ...
         repmat([min(ecg-mean(ecg))/2; max(ecg-mean(ecg))/2], size(qrs_i_raw)), ...
         'LineWidth', 2.5, 'LineStyle', '-.', 'Color', 'r');
    linkaxes(az, 'x');
    zoom on;
end

end % main function

%% ================================================================== %%
function xi_ref = parabolic_peak(seg, xi)
    if xi > 1 && xi < length(seg)
        alpha = double(seg(xi - 1));
        beta  = double(seg(xi));
        gamma = double(seg(xi + 1));
        denom = (alpha - 2*beta + gamma);
        if abs(denom) > eps
            p      = 0.5 * (alpha - gamma) / denom;
            p      = max(-0.5, min(0.5, p));   % clamp to ±0.5 samples
            xi_ref = xi + p;
        else
            xi_ref = xi;   % flat peak — no correction possible
        end
    else
        xi_ref = xi;       % edge of segment — no neighbours available
    end
end
