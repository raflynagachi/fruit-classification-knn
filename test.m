clc; clear; clearvars; close all; warning off all;

% list buah
fruits = ["apel" "jeruk" "tomat" "salak" "mangga"];
for fruit = 1:length(fruits)
    %%% Run through all fruits
    % membaca file citra
    nama_folder = 'dataset_test/' + fruits(fruit);
    nama_file = dir(fullfile(nama_folder,'*.jpg'));
    jumlah_file = numel(nama_file);
    
    % menginisiasi variabel ciri dan target
    ciri_warna = zeros(jumlah_file, 3);
    ciri_bentuk = zeros(jumlah_file, 2);
    target_warna = cell(jumlah_file, 1);
    
    % melakukan pengolahan citra terhadap seluruh file
    for n = 1:jumlah_file
        % membaca file rgb
        img = im2double(imread(fullfile(nama_folder, nama_file(n).name)));
        
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
        ciri_warna(n,1) = Red;
        ciri_warna(n,2) = Green;
        ciri_warna(n,3) = Blue;
        % target warna dengan nama buah
        target_warna{n} = fruits(fruit);

        %ciri bentuk
        stats = regionprops(bw,'Circularity','Eccentricity');
        circularity = stats.Circularity;
        eccentricity = stats.Eccentricity;

        ciri_bentuk(n,1) = circularity;
        ciri_bentuk(n,2) = eccentricity;

    end
    % menyusun variabel ciri_uji dan target_uji
    if exist('ciri_uji','var')==1
        ciri_uji = [ciri_uji; ciri_warna, ciri_bentuk];
        target_uji = [target_uji; transpose([target_warna{:}])];
    else
        ciri_uji = [ciri_warna,ciri_bentuk];
        target_uji = [transpose([target_warna{:}])];
    end
end

% memanggil model k-nn hasil pelatihan
load Mdl

% membaca kelas keluaran hasil pengujian
hasil_uji = predict(Mdl, ciri_uji);

jumlah_benar = 0;
jumlah_data = size(ciri_uji,1);
for k = 1:jumlah_data
    if isequal(hasil_uji{k},target_uji(k))
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pengujian = jumlah_benar/jumlah_data*100;