%% test_core_detection_pairing.m
% Unit test for core detection and pairing pipeline.
% Generates synthetic IQ data with known bubble positions, runs
% ULM_localization2D, ULM_tracking2D, and PALA_PairingAlgorithm,
% then saves all intermediate values for cross-validation with Python.
%
% Usage: run from the PALA/tests/ directory, or set paths manually.
% Output: test_detection_pairing_results.mat

%% Setup paths
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'PALA', 'PALA_addons')));

%% Parameters
rng(42); % Fixed seed for reproducibility

% Image grid
Nz = 50;  % height (axial, rows)
Nx = 40;  % width (lateral, columns)
NFrames = 30;

% PData - pixel data configuration (simplified)
PData.Origin = [0, 0, 0];        % [x, y, z] origin in wavelengths
PData.Size   = [Nz, Nx, 1];      % [rows, cols, pages]
PData.PDelta = [1, 0, 1];        % [dx, dy, dz] pixel spacing in wavelengths
% So pixel (iz, ix) maps to wavelength coords: z = (iz-1)*PDelta(3) + Origin(3), x = (ix-1)*PDelta(1) + Origin(1)

% ULM parameters
ULM = struct( ...
    'numberOfParticles', 10, ...  % Max detections per frame
    'res', 10, ...                % Resolution factor
    'max_linking_distance', 2, ...% In pixels, converted to wavelengths later
    'min_length', 3, ...          % Min track length (short for test)
    'fwhm', [3 3], ...            % FWHM of PSF [z x]
    'max_gap_closing', 0, ...     % No gap closing
    'size', [Nz, Nx, NFrames], ...
    'scale', [1 1 1], ...         % [z x t]
    'interp_factor', 1/10);

%% Generate synthetic bubble trajectories (ground truth)
% 5 bubbles moving in straight lines across the image
nBubbles = 5;
bubble_amplitude = 100;  % Peak intensity of each bubble
sigma_psf = 1.2;         % Gaussian PSF width (std dev)

% Starting positions [z, x] in pixel coords - keep away from boundaries
z_start = [10, 15, 25, 35, 40];
x_start = [8, 20, 30, 15, 25];

% Velocities [dz, dx] per frame in pixels
vz = [0.3, -0.2, 0.0, 0.1, -0.15];
vx = [0.2, 0.3, 0.25, -0.2, 0.1];

% Build ground truth positions: ListPos_ref [max_bubbles x 4 x NFrames]
% Format per frame: [x, y, z, reflectivity] (PALA convention)
ListPos_ref = nan(nBubbles, 4, NFrames);

% Also store ground truth in [z, x] wavelength coords for validation
gt_positions_wl = zeros(nBubbles, 2, NFrames); % [z, x] in wavelengths

for ifr = 1:NFrames
    for ib = 1:nBubbles
        z_pos = z_start(ib) + vz(ib) * (ifr - 1);
        x_pos = x_start(ib) + vx(ib) * (ifr - 1);

        % Check bounds (stay within image minus PSF border)
        if z_pos < 4 || z_pos > Nz-3 || x_pos < 4 || x_pos > Nx-3
            continue
        end

        % Convert pixel to wavelength coordinates
        z_wl = (z_pos - 1) * PData.PDelta(3) + PData.Origin(3);
        x_wl = (x_pos - 1) * PData.PDelta(1) + PData.Origin(1);

        % ListPos_ref format: [x, y, z, reflectivity]
        ListPos_ref(ib, :, ifr) = [x_wl, 0, z_wl, 1];
        gt_positions_wl(ib, :, ifr) = [z_wl, x_wl];
    end
end

%% Generate synthetic IQ images
IQ = zeros(Nz, Nx, NFrames, 'single');

% Create meshgrid for Gaussian PSF rendering
[zz, xx] = meshgrid(1:Nx, 1:Nz);  % Note: meshgrid(cols, rows) -> xx has rows, zz has cols

for ifr = 1:NFrames
    frame = zeros(Nz, Nx);
    for ib = 1:nBubbles
        z_pos = z_start(ib) + vz(ib) * (ifr - 1);
        x_pos = x_start(ib) + vx(ib) * (ifr - 1);

        if z_pos < 4 || z_pos > Nz-3 || x_pos < 4 || x_pos > Nx-3
            continue
        end

        % Add Gaussian blob: rows=z(axial), cols=x(lateral)
        blob = bubble_amplitude * exp(-((xx - z_pos).^2 + (zz - x_pos).^2) / (2 * sigma_psf^2));
        frame = frame + blob;
    end
    % Add small amount of noise for realism
    frame = frame + 0.5 * abs(randn(Nz, Nx));
    IQ(:,:,ifr) = single(frame);
end

%% ========== TEST 1: Detection / Localization ==========
fprintf('=== Test 1: ULM_localization2D ===\n');

% Test with two methods
test_methods = {'nolocalization', 'wa', 'radial'};
MatTracking_results = {};

for im = 1:numel(test_methods)
    ULM_test = ULM;
    ULM_test.LocMethod = test_methods{im};
    if strcmp(test_methods{im}, 'wa')
        ULM_test.parameters.NLocalMax = 2;
    elseif strcmp(test_methods{im}, 'radial')
        ULM_test.parameters.NLocalMax = 2;
    else
        ULM_test.parameters.NLocalMax = 2;
    end

    MatTracking = ULM_localization2D(IQ, ULM_test);
    MatTracking_results{im} = MatTracking;

    fprintf('  Method: %s -> %d detections\n', test_methods{im}, size(MatTracking, 1));
    fprintf('    Columns: [intensity, z_px, x_px, frame]\n');
    fprintf('    First 5 rows:\n');
    disp(MatTracking(1:min(5, size(MatTracking,1)), :));
end

%% ========== TEST 2: Coordinate conversion (pixel -> wavelength) ==========
fprintf('=== Test 2: Coordinate conversion ===\n');

% Use 'wa' results for tracking and pairing (index 2)
MatTracking_wa = double(MatTracking_results{2});

% Convert from pixel to wavelength (same as PALA_multiULM does)
MatTracking_wl = MatTracking_wa;
MatTracking_wl(:,2:3) = (MatTracking_wa(:,2:3) - [1 1]) .* PData.PDelta([3 1]) + [PData.Origin(3) PData.Origin(1)];

fprintf('  Converted %d detections to wavelength coords\n', size(MatTracking_wl, 1));

%% ========== TEST 3: Tracking ==========
fprintf('=== Test 3: ULM_tracking2D ===\n');

ULM_track = ULM;
% Convert max_linking_distance to wavelength (same as PALA_multiULM)
ULM_track.max_linking_distance = ULM.max_linking_distance * PData.PDelta(3);

[Tracks_raw, Tracks_interp] = ULM_tracking2D(MatTracking_wl, ULM_track, 'pala');

fprintf('  Found %d raw tracks\n', numel(Tracks_raw));
for it = 1:min(5, numel(Tracks_raw))
    fprintf('    Track %d: %d points\n', it, size(Tracks_raw{it}, 1));
end

%% ========== TEST 4: Pairing Algorithm ==========
fprintf('=== Test 4: PALA_PairingAlgorithm ===\n');

% Build MatTrackedLoc from raw tracks: [z, x, frame]
MatTrackedLoc = cell2mat(Tracks_raw);

Threshold_pairing = 2.0;  % wavelengths (generous for synthetic data)
Threshold_TruePos = 1.0;  % wavelengths

[Stat_classification, ErrList, FinalPairs, MissingPoint, WrongLoc] = ...
    PALA_PairingAlgorithm(ListPos_ref, MatTrackedLoc, PData, Threshold_pairing, Threshold_TruePos);

fprintf('  Stat_classification (per frame sum):\n');
fprintf('    Total: Npos_in=%d, Npos_loc=%d, T_pos=%d, F_neg=%d, F_pos=%d\n', ...
    sum(Stat_classification(:,1)), sum(Stat_classification(:,2)), ...
    sum(Stat_classification(:,3)), sum(Stat_classification(:,4)), ...
    sum(Stat_classification(:,5)));

if ~isempty(ErrList)
    fprintf('  Mean localization error (norm): %.4f wavelengths\n', mean(ErrList(:,1)));
    fprintf('  Std localization error (norm): %.4f wavelengths\n', std(ErrList(:,1)));
end

%% ========== TEST 5: Also run nolocalization for comparison ==========
fprintf('=== Test 5: Full pipeline with nolocalization ===\n');

MatTracking_noloc = double(MatTracking_results{1});
MatTracking_noloc_wl = MatTracking_noloc;
MatTracking_noloc_wl(:,2:3) = (MatTracking_noloc(:,2:3) - [1 1]) .* PData.PDelta([3 1]) + [PData.Origin(3) PData.Origin(1)];

ULM_track_noloc = ULM_track;
[Tracks_raw_noloc, ~] = ULM_tracking2D(MatTracking_noloc_wl, ULM_track_noloc, 'pala');

MatTrackedLoc_noloc = cell2mat(Tracks_raw_noloc);

[Stat_noloc, ErrList_noloc, FinalPairs_noloc, MissingPoint_noloc, WrongLoc_noloc] = ...
    PALA_PairingAlgorithm(ListPos_ref, MatTrackedLoc_noloc, PData, Threshold_pairing, Threshold_TruePos);

fprintf('  nolocalization: T_pos=%d, F_neg=%d, F_pos=%d\n', ...
    sum(Stat_noloc(:,3)), sum(Stat_noloc(:,4)), sum(Stat_noloc(:,5)));

%% ========== TEST 6: Radial symmetry pipeline ==========
fprintf('=== Test 6: Full pipeline with radial ===\n');

MatTracking_rad = double(MatTracking_results{3});
MatTracking_rad_wl = MatTracking_rad;
MatTracking_rad_wl(:,2:3) = (MatTracking_rad(:,2:3) - [1 1]) .* PData.PDelta([3 1]) + [PData.Origin(3) PData.Origin(1)];

[Tracks_raw_rad, ~] = ULM_tracking2D(MatTracking_rad_wl, ULM_track, 'pala');

MatTrackedLoc_rad = cell2mat(Tracks_raw_rad);

[Stat_rad, ErrList_rad, FinalPairs_rad, MissingPoint_rad, WrongLoc_rad] = ...
    PALA_PairingAlgorithm(ListPos_ref, MatTrackedLoc_rad, PData, Threshold_pairing, Threshold_TruePos);

fprintf('  radial: T_pos=%d, F_neg=%d, F_pos=%d\n', ...
    sum(Stat_rad(:,3)), sum(Stat_rad(:,4)), sum(Stat_rad(:,5)));

%% ========== Save all results for Python comparison ==========
fprintf('\n=== Saving results ===\n');

results_file = fullfile(fileparts(mfilename('fullpath')), 'test_detection_pairing_results.mat');

save(results_file, ...
    ... % Input data
    'IQ', 'Nz', 'Nx', 'NFrames', 'nBubbles', 'bubble_amplitude', 'sigma_psf', ...
    'z_start', 'x_start', 'vz', 'vx', ...
    ... % Configuration
    'PData', 'ULM', 'Threshold_pairing', 'Threshold_TruePos', ...
    'ListPos_ref', 'gt_positions_wl', ...
    ... % Detection results (pixel coords)
    'MatTracking_results', 'test_methods', ...
    ... % WA pipeline results
    'MatTracking_wl', ...
    'Tracks_raw', 'Tracks_interp', ...
    'MatTrackedLoc', ...
    'Stat_classification', 'ErrList', 'FinalPairs', 'MissingPoint', 'WrongLoc', ...
    ... % Nolocalization pipeline results
    'MatTracking_noloc_wl', ...
    'Tracks_raw_noloc', 'MatTrackedLoc_noloc', ...
    'Stat_noloc', 'ErrList_noloc', 'FinalPairs_noloc', ...
    ... % Radial pipeline results
    'MatTracking_rad_wl', ...
    'Tracks_raw_rad', 'MatTrackedLoc_rad', ...
    'Stat_rad', 'ErrList_rad', 'FinalPairs_rad', ...
    '-v7');

fprintf('Results saved to: %s\n', results_file);
fprintf('\nAll tests completed.\n');
