clear,close all;

% folder = 'CBSD68_mod\';
% folder = 'DIV2K_train_HR_sub\';
% folder = 'DIV2K100\DIV2K100_mod2\';
folder = 'LIVE1\LIVE1_mod\';
% modfolder = 'BSD100\Set12_rgb_mod\';
% savefolder = 'LIVE1\LIVE1_gray_jpg80\';
% savefolder = 'DIV2K800_sub_jpg50\';
% savefolder = 'CBSD68_Gaussian55\';
% modfolder = 'Classic5\classic5_mod\';
savefolder = 'LIVE1\LIVE1_jpg50\';
% savefolder = 'DIV2K100\DIV2K100_sub_jpg30\';
% noiseSigma = 55;
filetype = '.bmp';
% JPEG_Quality = 50;

filepaths = dir(fullfile(folder, ['*' filetype]));
num_imgs = length(filepaths);

% 
% random seed for generating training data
for i = 1:num_imgs
	[imaddress,imname,type] = fileparts(filepaths(i).name);
	im = imread([folder filepaths(i).name]);
	% randn('seed', 0);
	% if size(im, 3) == 1
	% 	[h, w, c] = size(im);
	% 	im_rgb = zeros(h, w, 3, 'uint8');
	% 	channel = im(:,:,1);
 %        im_rgb(:,:,1) = channel;
 %        im_rgb(:,:,2) = channel;
 %        im_rgb(:,:,3) = channel;
 %        imwrite(im_rgb, [savefolder imname filetype]);
 %    end

 %    im = modcrop(im, 2);
	% imwrite(im, [modfolder imname filetype]);

	im = im2double(im);
	
    noise = noiseSigma/255.*randn(size(im));
    im_noise = single(im + noise);
    im_noise = im2uint8(im_noise);
	% % % im = rgb2ycbcr(im);
	% % % imwrite(im(:,:,1), [savefolder_gray imname filetype]);
	imwrite(im_noise, [savefolder imname filetype]);
	% imwrite(im, [savefolder imname '_JPEG' '.jpg'], 'jpg', 'Quality', JPEG_Quality);
end



