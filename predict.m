clc; clear; close all; warning off all;

% panggil menu "browse file"
[nama_file, nama_folder] = uigetfile('*.jpg');

% jika ada nama file yang dipilih maka ekseskusi
if ~isequal(nama_file,0)
    % membaca file citra rgb
    img = im2double(imread(fullfile(nama_folder, nama_file)));
    % konversi citra rgb menjadi grayscale
    img_gray = rgb2gray(img);
    %%% figure, imshow(img);
    %%% figure, imshow(img_gray);
    
    % mengkonversi citra grayscale menjadi biner
    bw= imbinarize(img_gray, 0.5);
    %%% figure, imshow(bw);
    
    % komplemen citra
    bw = imcomplement(bw);
    %%% figure, imshow(bw);
    
    % operasi morfologi untuk menyempurnakan segmentasi
    % 1.filling holes
    bw = imfill(bw, 'holes');
    %%% figure, imshow(bw);
    % 2. area opening
    bw = bwareaopen(bw, 100);
    %%% figure, imshow(bw);
    % 3. Ekstraksi ciri
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    R(~bw) = 0;
    G(~bw) = 0;
    B(~bw) = 0;
    RGB = cat(3, R,G,B);
    %%% figure, imshow(RGB);
    % rata-rata warna RGB
    Red = sum(sum(R))/sum(sum(bw));
    Green = sum(sum(G))/sum(sum(bw));
    Blue = sum(sum(B))/sum(sum(bw));

    %ciri bentuk
    stats = regionprops(bw,'Circularity','Eccentricity');
    circularity = stats.Circularity;
    eccentricity = stats.Eccentricity;

    % menyusun variabel ciri_uji
    ciri_uji = [Red, Green, Blue, circularity, eccentricity];

    % panggil model k-nn hasil pelatihan
    load Mdl
    hasil_uji = predict(Mdl, ciri_uji);

    % tampilkan citra asli dan kelas prediksi
    figure, imshow(img, 'InitialMagnification','fit')
    title({['Nama File: ', nama_file], ['Kelas Prediksi: ', hasil_uji{1}]})
    
    figure, imshow(bw, 'InitialMagnification','fit')
    title({['Nama File: ', nama_file], ['Kelas Prediksi: ', hasil_uji{1}]})

else
    % jika tidak ada file yang dipilih
    return
end