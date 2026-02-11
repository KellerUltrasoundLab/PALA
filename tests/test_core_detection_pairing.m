%% test_core_detection_pairing.m
% Unit test for core detection and pairing pipeline using real InSilicoFlow data.
% Loads ONE IQ file from the PALA InSilicoFlow dataset, runs detection (ULM_localization2D),
% tracking (ULM_tracking2D), and pairing (PALA_PairingAlgorithm) for two localization
% methods ('wa' and 'radial'). Computes Accuracy and F1 score. Saves compact results
% for cross-validation with Python.
%
% Before running:
%   1. Download InSilicoFlow data from https://doi.org/10.5281/zenodo.4343435
%   2. Set PALA_data_folder below to point to where PALA_data_InSilicoFlow lives
%
% Output: tests/test_detection_pairing_results.mat (compact, ~few MB)

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'PALA', 'PALA_addons')));

%% ===== USER CONFIG =====
% Point this to the parent of PALA_data_InSilicoFlow
PALA_data_folder = fullfile(script_dir, '..');
% If your data lives elsewhere, override:
% PALA_data_folder = 'D:\PALA_test';
% =========================

workingdir = fullfile(PALA_data_folder, 'PALA_data_InSilicoFlow');
filename = 'PALA_InSilicoFlow';
myfilepath = fullfile(workingdir, filename);
myfilepath_data = fullfile(workingdir, 'IQ', filename);

fprintf('=== Loading sequence config ===\n');
listVar = {'P','PData','Trans','Media','UF','Resource','Receive','filetitle'};
load([myfilepath '_sequence.mat'], '-mat', listVar{:});

%% ULM parameters (same as PALA_SilicoFlow.m)
NFrames = P.BlocSize * P.numBloc;
framerate = P.FrameRate;
res = 10;

ULM = struct('numberOfParticles', 40, ...
    'res', 10, ...
    'max_linking_distance', 2, ...
    'min_length', 15, ...
    'fwhm', [3 3], ...
    'max_gap_closing', 0, ...
    'size', [PData.Size(1), PData.Size(2), NFrames], ...
    'scale', [1 1 1/framerate], ...
    'numberOfFramesProcessed', NFrames, ...
    'interp_factor', 1/res);

%% Load ONE IQ file (bloc 1) - this is the core speedup: single file instead of all blocs x all noise levels
fprintf('=== Loading single IQ file (bloc 001) ===\n');
iq_file = [myfilepath_data '_IQ001.mat'];
temp = load(iq_file, 'IQ', 'Media', 'ListPos');
IQ_raw = abs(temp.IQ);
ListPos = temp.ListPos;
fprintf('  IQ size: [%d x %d x %d]\n', size(IQ_raw,1), size(IQ_raw,2), size(IQ_raw,3));
fprintf('  ListPos size: [%d x %d x %d]\n', size(ListPos,1), size(ListPos,2), size(ListPos,3));

%% Apply noise at ONE clutter level (-60 dB = nearly clean, fast test)
NoiseParam.Power        = -2;
NoiseParam.Impedance    = .2;
NoiseParam.SigmaGauss   = 1.5;
NoiseParam.clutterdB    = -60;  % Low noise for cleaner test
NoiseParam.amplCullerdB = 10;

fprintf('=== Adding noise (clutter = %d dB) ===\n', NoiseParam.clutterdB);
IQ = PALA_AddNoiseInIQ(IQ_raw, NoiseParam);

%% Run detection + tracking for two algorithms: 'wa' and 'radial'
listAlgo = {'wa', 'radial'};
Nalgo = numel(listAlgo);

Threshold_pairing = 0.5;   % wavelengths (same as PALA_SilicoFlow.m)
Threshold_TruePos = 0.25;  % wavelengths (same as PALA_SilicoFlow.m)

% Store all pipeline outputs per algorithm
results = struct();

for ialgo = 1:Nalgo
    algoName = listAlgo{ialgo};
    fprintf('\n=== Pipeline: %s ===\n', algoName);

    %% 1. Detection & Localization
    ULM_run = ULM;
    switch lower(algoName)
        case 'wa'
            ULM_run.LocMethod = 'wa';
        case 'radial'
            ULM_run.LocMethod = 'radial';
    end
    ULM_run.parameters.NLocalMax = 3;

    t0 = tic;
    MatTracking_px = ULM_localization2D(IQ, ULM_run);
    det_time = toc(t0);
    fprintf('  Detection: %d particles in %.2f s\n', size(MatTracking_px,1), det_time);

    %% 2. Coordinate conversion (pixel -> wavelength, same as PALA_multiULM)
    MatTracking_wl = double(MatTracking_px);
    MatTracking_wl(:,2:3) = (double(MatTracking_px(:,2:3)) - [1 1]) .* PData.PDelta([3 1]) + [PData.Origin(3) PData.Origin(1)];

    %% 3. Tracking
    ULM_track = ULM;
    ULM_track.max_linking_distance = ULM.max_linking_distance * PData.PDelta(3);

    t0 = tic;
    [Tracks_raw, Tracks_interp] = ULM_tracking2D(MatTracking_wl, ULM_track, 'pala');
    track_time = toc(t0);
    fprintf('  Tracking: %d tracks in %.2f s\n', numel(Tracks_raw), track_time);

    %% 4. Pairing
    MatTrackedLoc = cell2mat(Tracks_raw);  % [z, x, frame]
    fprintf('  MatTrackedLoc: %d tracked points\n', size(MatTrackedLoc,1));

    t0 = tic;
    [Stat_classification, ErrList, FinalPairs, MissingPoint, WrongLoc] = ...
        PALA_PairingAlgorithm(ListPos, MatTrackedLoc, PData, Threshold_pairing, Threshold_TruePos);
    pair_time = toc(t0);

    %% 5. Compute metrics
    total_Npos_in  = sum(Stat_classification(:,1));
    total_Npos_loc = sum(Stat_classification(:,2));
    total_TP       = sum(Stat_classification(:,3));
    total_FN       = sum(Stat_classification(:,4));
    total_FP       = sum(Stat_classification(:,5));

    Precision  = total_TP / (total_TP + total_FP);
    Recall     = total_TP / (total_TP + total_FN);  % Sensitivity
    F1         = 2 * Precision * Recall / (Precision + Recall);
    Accuracy   = total_TP / (total_TP + total_FP + total_FN);  % Jaccard index
    MeanErr    = mean(ErrList(:,1));
    StdErr     = std(ErrList(:,1));

    fprintf('  Pairing done in %.2f s\n', pair_time);
    fprintf('  --- Results for %s ---\n', algoName);
    fprintf('  TP=%d, FN=%d, FP=%d\n', total_TP, total_FN, total_FP);
    fprintf('  Precision:  %.4f\n', Precision);
    fprintf('  Recall:     %.4f\n', Recall);
    fprintf('  F1 Score:   %.4f\n', F1);
    fprintf('  Accuracy (Jaccard): %.4f\n', Accuracy);
    fprintf('  Mean loc error: %.4f wl, Std: %.4f wl\n', MeanErr, StdErr);

    %% Store results
    results(ialgo).algoName = algoName;
    results(ialgo).MatTracking_px = MatTracking_px;
    results(ialgo).MatTracking_wl = MatTracking_wl;
    results(ialgo).Tracks_raw = Tracks_raw;
    results(ialgo).MatTrackedLoc = MatTrackedLoc;
    results(ialgo).Stat_classification = Stat_classification;
    results(ialgo).ErrList = ErrList;
    results(ialgo).FinalPairs = FinalPairs;
    results(ialgo).MissingPoint = MissingPoint;
    results(ialgo).WrongLoc = WrongLoc;
    results(ialgo).Precision = Precision;
    results(ialgo).Recall = Recall;
    results(ialgo).F1 = F1;
    results(ialgo).Accuracy = Accuracy;
    results(ialgo).MeanErr = MeanErr;
    results(ialgo).StdErr = StdErr;
end

%% ===== Save compact results for Python replication =====
fprintf('\n=== Saving results ===\n');

% Save a SMALL subset of IQ (first 10 frames) for Python replication of detection
IQ_subset = IQ(:,:,1:min(10, size(IQ,3)));
IQ_subset_nframes = size(IQ_subset, 3);

% Save the corresponding ListPos subset
ListPos_subset = ListPos(:,:,1:min(10, size(ListPos,3)));

results_file = fullfile(script_dir, 'test_detection_pairing_results.mat');

save(results_file, ...
    'IQ_subset', 'IQ_subset_nframes', ...      % Small IQ for Python detection test
    'ListPos', 'ListPos_subset', ...            % Ground truth
    'PData', 'ULM', ...                         % Configuration
    'NoiseParam', 'Threshold_pairing', 'Threshold_TruePos', ...
    'listAlgo', 'results', ...                  % All per-algorithm results
    '-v7');

% Also save full IQ separately (larger file, only needed if Python wants full replication)
full_results_file = fullfile(script_dir, 'test_detection_pairing_full_IQ.mat');
save(full_results_file, 'IQ', 'ListPos', 'PData', 'ULM', ...
    'NoiseParam', 'Threshold_pairing', 'Threshold_TruePos', ...
    'listAlgo', 'results', '-v7');

fprintf('Compact results saved to: %s\n', results_file);
fprintf('Full IQ results saved to: %s\n', full_results_file);

%% Print final summary
fprintf('\n========== FINAL SUMMARY ==========\n');
for ialgo = 1:Nalgo
    fprintf('  %s: Accuracy(Jaccard)=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f\n', ...
        results(ialgo).algoName, results(ialgo).Accuracy, results(ialgo).F1, ...
        results(ialgo).Precision, results(ialgo).Recall);
end
fprintf('====================================\n');
