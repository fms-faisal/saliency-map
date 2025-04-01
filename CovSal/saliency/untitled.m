%% Clear workspace
clear all;
clc;

fprintf('Starting EMD evaluation...\n');

% Define folders
td_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\TD_FixMaps';
asd_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\ASD_FixMaps';
prediction_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency_Maps';

fprintf('Loading prediction files...\n');
% Get list of prediction files
prediction_files = dir(fullfile(prediction_folder, 'Saliency_*.png'));
num_images = length(prediction_files);

fprintf('Initializing EMD storage...\n');
% Initialize EMD storage
emd_td = zeros(num_images, 1);
emd_asd = zeros(num_images, 1);

fprintf('Processing images...\n');
% Loop through each image
for i = 1:num_images
    % Extract image number from prediction file name
    file_name = prediction_files(i).name;
    start_idx = find(file_name == '_') + 1;
    end_idx = find(file_name == '.') - 1;
    image_num = str2double(file_name(start_idx:end_idx));
    
    fprintf('Processing image %d of %d (Image number: %d)\n', i, num_images, image_num);
    
    % Load prediction map
    prediction_path = fullfile(prediction_folder, file_name);
    prediction_map = imread(prediction_path);
    prediction_map = im2double(prediction_map);
    
    % Load TD fixation map
    td_fixation_file = sprintf('%d_s.png', image_num);
    td_fixation_path = fullfile(td_fixation_folder, td_fixation_file);
    td_fixation_map = imread(td_fixation_path);
    td_fixation_map = im2double(td_fixation_map);
    
    % Load ASD fixation map
    asd_fixation_file = sprintf('%d_s.png', image_num);
    asd_fixation_path = fullfile(asd_fixation_folder, asd_fixation_file);
    asd_fixation_map = imread(asd_fixation_path);
    asd_fixation_map = im2double(asd_fixation_map);
    
    % Resize all maps to a common size (e.g., 256x256) for consistency
    target_size = [256, 256];
    prediction_map = imresize(prediction_map, target_size);
    td_fixation_map = imresize(td_fixation_map, target_size);
    asd_fixation_map = imresize(asd_fixation_map, target_size);
    
    % Compute EMD for TD
    fprintf('  Computing EMD for TD...\n');
    emd_td(i) = compute_wasserstein_distance(prediction_map, td_fixation_map);
    
    % Compute EMD for ASD
    fprintf('  Computing EMD for ASD...\n');
    emd_asd(i) = compute_wasserstein_distance(prediction_map, asd_fixation_map);
    
    fprintf('  Image %d processed successfully\n\n', image_num);
end

fprintf('Computing mean EMD...\n');
% Compute mean EMD
mean_emd_td = mean(emd_td);
mean_emd_asd = mean(emd_asd);

fprintf('Displaying results...\n');
% Display results
fprintf('\nEMD Evaluation Results:\n');
fprintf('EMD\t\t\tTD\t\tASD\n');
fprintf('EMD\t\t\t%.4f\t%.4f\n', mean_emd_td, mean_emd_asd);

%% Wasserstein Distance Calculation Function
function distance = compute_wasserstein_distance(saliency_map, fixation_map)
    % Reduce image size for efficiency
    downsize = 16;
    im1 = imresize(fixation_map, 1/downsize);
    im2 = imresize(saliency_map, size(im1));
    
    % Normalize maps to sum to 1
    im1 = im1 / sum(im1(:));
    im2 = im2 / sum(im2(:));
    
    % Flatten the images into 1D histograms
    P = double(im1(:));
    Q = double(im2(:));
    
    % Compute Wasserstein distance (simplified EMD)
    distance = wasserstein_distance(P, Q);
    
    fprintf('      EMD computed: %.4f\n', distance);
end

%% Wasserstein Distance Calculation
function distance = wasserstein_distance(P, Q)
    % Compute cumulative distribution functions
    cdf_P = cumsum(P);
    cdf_Q = cumsum(Q);
    
    % Compute Wasserstein distance
    distance = sum(abs(cdf_P - cdf_Q));
end