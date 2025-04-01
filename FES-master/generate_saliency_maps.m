clear all;
clc;

%% Define folder paths
image_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\Images';
output_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\fes_Saliency_Maps';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Load prior (if available)
load('prior'); % Ensure the prior file exists

%% Get list of images
image_files = dir(fullfile(image_folder, '*.png'));
num_images = length(image_files);

%% Process each image
for i = 1:num_images
    % Read the original image
    image_name = image_files(i).name;
    image_path = fullfile(image_folder, image_name);
    original_image = imread(image_path);

    % Get image dimensions
    [x, y, ~] = size(original_image);

    % Convert to LAB color space
    lab_image = RGB2Lab(original_image);

    % Compute saliency map
    saliency_map = computeFinalSaliency(lab_image, [8 8 8], [13 25 38], 30, 10, 1, p1);
    saliency_map = imresize(saliency_map, [x, y]);

    % Save the saliency map
    saliency_map_name = sprintf('Saliency_%s', image_name);
    saliency_map_path = fullfile(output_folder, saliency_map_name);
    imwrite(saliency_map, saliency_map_path);

    fprintf('Generated saliency map for image %s\n', image_name);
end

%% Helper function: Convert RGB to LAB color space
function lab_image = RGB2Lab(rgb_image)
    % Convert RGB image to LAB color space
    cform = makecform('srgb2lab');
    lab_image = applycform(rgb_image, cform);
end