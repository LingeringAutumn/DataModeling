clc; clear;

% 读取的碎纸片总数，编号从 000 到 018
num_pieces = 19;

% 分别用于存放左边缘、右边缘、左边缘黑色像素数
left_edges = cell(num_pieces, 1);
right_edges = cell(num_pieces, 1);
black_left_count = zeros(num_pieces, 1);

for i = 0:num_pieces-1
    % 构造图像文件名，比如 000.bmp
    filename = sprintf('%03d.bmp', i);

    % 读取图像，若为彩色图则转为灰度图
    img = imread(filename);
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % 简单二值化处理
    bw = imbinarize(gray, 0.5);

    % 提取图像的左右边缘
    left_edges{i+1} = bw(:, 1);
    right_edges{i+1} = bw(:, end);

    % 统计左边缘为黑色的像素数量，用于判断起始碎片
    black_left_count(i+1) = sum(bw(:, 1) == 0);
end

% 从左边缘最空白的碎片开始作为起点
[~, start_idx] = min(black_left_count);
used = false(num_pieces, 1);
used(start_idx) = true;
sequence = start_idx;

% 初始化误差矩阵，每一项表示右边对左边的边缘匹配差值
match_error = inf(num_pieces);

for i = 1:num_pieces
    for j = 1:num_pieces
        if i ~= j
            % 计算两个边缘之间的像素差，使用 L1 距离
            match_error(i,j) = sum(abs(double(right_edges{i}) - double(left_edges{j})));
        end
    end
end

% 按照误差最小的方式依次选择下一个碎片（贪心）
current = start_idx;
while sum(used) < num_pieces
    candidates = match_error(current, :);
    candidates(used) = inf;
    [~, next] = min(candidates);

    sequence(end+1) = next;
    used(next) = true;
    current = next;
end

% 打印拼接顺序
disp('最终拼接顺序如下（碎片编号从000开始）：');
disp(sequence - 1);

% 依次拼接图像并显示
stitched = [];
for i = sequence
    img = imread(sprintf('%03d.bmp', i-1));
    stitched = [stitched, img];
end

figure;
imshow(stitched);
title('自动拼接结果');

% 将拼接顺序保存到文本文件
fid = fopen('result_problem1.txt', 'w');
fprintf(fid, '附件1拼接顺序（从左到右）：\n');
fprintf(fid, '%03d ', sequence - 1);
fclose(fid);
