% This is a matlab code
% The Z-score outlier detection method is used to identify the abnormal parts on the three channel color image.

% Read color image
image = imread('3-1.jpg');

% Convert image to double type
double_image = im2double(image);

% Calculate Z-score for each channel
z_scores = zeros(size(double_image));
for channel = 1:3
    channel_data = double_image(:, :, channel);
    mean_channel = mean(channel_data(:));
    std_channel = std(channel_data(:));
    z_scores(:, :, channel) = abs((channel_data - mean_channel) / std_channel);
end

% Calculate the overall Z-score
total_z_scores = sum(z_scores, 3);

% Set threshold to identify abnormal parts
threshold = 3.0; % Adjustable threshold
outliers = total_z_scores > threshold;

% Mark the abnormal part on the original image
marked_image = image;
marked_image(repmat(outliers, [1, 1, 3])) = 255; % Mark the abnormal part as white

% Display results
figure;
subplot(2, 1, 1);
imshow(image);
title('original image');

subplot(2, 1, 2);
imshow(marked_image);
title('Mark abnormal part');

% Save the result
imwrite(marked_image, 'marked_image_3-3.jpg');
