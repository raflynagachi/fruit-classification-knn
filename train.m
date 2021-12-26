clc; clear; clearvars; close all; warning off all;

% list buah
fruits_scalar = ["apel" "jeruk" "tomat" "salak" "mangga"];
for fruit = 1:length(fruits_scalar)
    %%% Run through all fruits
    % membaca file citra
    nama_folder = 'dataset_train/' + fruits_scalar(fruit);
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
    
        % mengkonversi citra grayscale menjadi biner
        bw= imbinarize(img_gray, 0.5);
    
        % komplemen citra
        bw = imcomplement(bw);
    
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
        name_fruit = fruits_scalar(fruit);
        target_warna{n} = name_fruit;

        %ciri bentuk
        stats = regionprops(bw,'Circularity','Eccentricity');
        circularity = stats.Circularity;
        eccentricity = stats.Eccentricity;

        ciri_bentuk(n,1) = circularity;
        ciri_bentuk(n,2) = eccentricity;

    end
    % menyusun variabel ciri_latih dan target_latih
    if exist('ciri_latih','var')==1
        ciri_latih = [ciri_latih; ciri_warna,ciri_bentuk];
        target_latih = [target_latih; transpose([target_warna{:}])];
    else
        ciri_latih = [ciri_warna,ciri_bentuk];
        target_latih = [transpose([target_warna{:}])];
    end
end

% pelatihan menggunakan k-nn
Mdl = fitcknn(ciri_latih, target_latih,'NumNeighbors',5);

% membaca kelas keluaran hasil pelatihan
hasil_latih = predict(Mdl, ciri_latih);

jumlah_benar = 0;
jumlah_data = size(ciri_latih,1);
for k = 1:jumlah_data
    if isequal(hasil_latih{k},target_latih(k))
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pelatihan = jumlah_benar/jumlah_data*100;

%simpan variable Mdl hasil pelatihan
save Mdl Mdl