import cupy as cp
import numpy as np

class PerspectiveTransformer:
    def __init__(self, batch_size=11, height=3000, width=4096, height_cropped=2720, width_cropped=3840):
        """初始化透视变换器"""
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.height_cropped = height_cropped
        self.width_cropped = width_cropped
        self.height_crop_size = cp.int32((self.height - self.height_cropped) / 2)
        self.width_crop_size = cp.int32((self.width - self.width_cropped) / 2)

        # CUDA 设置
        self.block_size = (16, 16)
        self.grid_size = (
            (width + self.block_size[0] - 1) // self.block_size[0],
            (height + self.block_size[1] - 1) // self.block_size[1],
            batch_size
        )

        # 编译CUDA kernel
        self.kernel = self.get_transform_kernel()

        # 初始化GPU内存
        self.initialize_gpu_memory()

    def initialize_gpu_memory(self):
        """初始化GPU内存"""
        self.d_output = cp.zeros((self.batch_size, self.height_cropped, self.width_cropped),
                                 dtype=cp.float32)

    def get_transform_kernel(self):
        """获取CUDA kernel"""
        kernel_code = r'''
        extern "C" __global__ void perspective_transform(
            unsigned char* __restrict__ src,
            float* __restrict__ dst,
            const float* __restrict__ matrices,
            const int batch_size,
            const int height,
            const int width,
            const int height_cropped,
            const int width_cropped,
            const int height_crop_size,
            const int width_crop_size
        ) {
            __shared__ float sh_matrix[9];

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int x = blockIdx.x * blockDim.x + tx; // 这里的 x 和 y 对应的都是未裁剪的坐标  
            const int y = blockIdx.y * blockDim.y + ty; // x in [0,4096], y in [0, 3000]
            const int batch_idx = blockIdx.z;
            const float eps = 1e-8f;

            // 加载变换矩阵到共享内存
            if (ty < 3 && tx < 3) {
                if (batch_idx < 8){
                    sh_matrix[ty * 3 + tx] = matrices[batch_idx * 9 + ty * 3 + tx];
                    }
            }
            __syncthreads();

            // 基准图像裁剪范围之外的坐标对应的区域不参与计算
            if (x < width_crop_size || x > width-width_crop_size || y < height_crop_size || y > height-height_crop_size || batch_idx >= batch_size)
                return; 

            // 计算源坐标，源坐标对应的是被配准图像的
            const float w = sh_matrix[6] * x + sh_matrix[7] * y + sh_matrix[8];

            if (abs(w) > 1e-8f) {
            // 计算被透视变换图像对应坐标点
                const float src_x = (sh_matrix[0] * x + sh_matrix[1] * y + sh_matrix[2]) / w; 
                const float src_y = (sh_matrix[3] * x + sh_matrix[4] * y + sh_matrix[5]) / w;

                // 检查坐标是否在有效范围内并执行双线性插值
                if (src_x >= 0 && src_x < width - 1 && src_y >= 0 && src_y < height - 1) {
                    const int x0 = __float2int_rd(src_x);   // 取整数
                    const int y0 = __float2int_rd(src_y);
                    const int x1 = x0 + 1;
                    const int y1 = y0 + 1;

                    const float wx1 = src_x - x0;  //计算插值权重
                    const float wy1 = src_y - y0;
                    const float wx0 = 1.0f - wx1;
                    const float wy0 = 1.0f - wy1;


                    if (batch_idx <= 7){
                        const int src_offset = batch_idx * height * width;
                        const int dst_idx = batch_idx * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);

                        dst[dst_idx] = 
                            src[src_offset + y0 * width + x0] * wx0 * wy0 +
                            src[src_offset + y0 * width + x1] * wx1 * wy0 +
                            src[src_offset + y1 * width + x0] * wx0 * wy1 +
                            src[src_offset + y1 * width + x1] * wx1 * wy1;

                    }else if (batch_idx == 8){

                        const int dst_idx_ndvi = batch_idx * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_red = 6 * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_nir = 5 * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);

                        dst[dst_idx_ndvi] = 122.5f + 122.5f * (dst[dst_idx_nir] - dst[dst_idx_red])/(dst[dst_idx_nir] + dst[dst_idx_red] + eps);

                    }else if (batch_idx == 9){

                        const int dst_idx_dolp = batch_idx * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_aolp = (batch_idx + 1) * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_i0 = (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_i60 = height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);
                        const int dst_idx_i120 = 2 * height_cropped * width_cropped + (y-height_crop_size-1) * (width_cropped) + (x-width_crop_size-1);

                        const float I = 2.0f/3.0f * (dst[dst_idx_i0] + dst[dst_idx_i60] + dst[dst_idx_i120]);
                        const float Q = 2.0f/3.0f * (2.0f * dst[dst_idx_i0] - dst[dst_idx_i60] - dst[dst_idx_i120]);
                        const float U = 2.0f/sqrtf(3.0f) * (dst[dst_idx_i60] - dst[dst_idx_i120]);

                        dst[dst_idx_dolp] = 255.0f * sqrtf((Q * Q + U * U) / (I * I + eps));
                        dst[dst_idx_aolp] = 122.5f + (255.0 / 3.1415926f) * (0.5f * atan2f(U, Q + eps));

                    }
                }
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'perspective_transform')

    def process(self, cpu_images, cpu_matrices):
        # # [11, 2720, 3840]
        """执行透视变换并返回GPU结果

        Args:
            cpu_images: numpy array (batch_size, height, width)

        Returns:
            cupy array (batch_size, height, width)
        """
        imgs = np.zeros((8, 3000, 4096), dtype=np.uint8)
        for ii in range(8):
            key_Cam = 'Cam' + str(ii)
            imgs[ii, :, :] = cpu_images[key_Cam]

        gpu_images = cp.asarray(imgs)
        gpu_matrices = cp.asarray(cpu_matrices)

        self.kernel(
            self.grid_size,
            self.block_size,
            (
                gpu_images,
                self.d_output.ravel(),
                gpu_matrices.ravel(),
                self.batch_size,
                self.height,
                self.width,
                self.height_cropped,
                self.width_cropped,
                self.height_crop_size,
                self.width_crop_size
            )
        )
        # return gpu_images
        return self.d_output



class DownSampler:
    def __init__(self, batch_size=10, input_height=2720, input_width=3840, resize_height=340, resize_width=480):
        """初始化图像处理器

        Args:
            batch_size: 批处理大小
            input_height: 输入图像高度
            input_width: 输入图像宽度
            resize_height: MOG2处理的目标高度
            resize_width: MOG2处理的目标宽度
        """
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = resize_height
        self.output_width = resize_width

        # CUDA设置
        self.block_size = (16, 16)
        self.grid_size = (
            (self.output_width + self.block_size[0] - 1) // self.block_size[0],
            (self.output_height + self.block_size[1] - 1) // self.block_size[1],
            self.batch_size
        )

        # 初始化下采样kernel
        self.downsample_kernel = self.get_downsample_kernel()

        # GPU内存管理
        self.initialize_gpu_memory()

    def initialize_gpu_memory(self):
        """初始化并预分配GPU内存"""
        # 为下采样结果分配内存
        self.d_downsampled = cp.zeros((self.batch_size, self.output_height,
                                       self.output_width), dtype=cp.float32)

    def get_downsample_kernel(self):
        """获取下采样CUDA kernel"""
        kernel_code = r'''
        extern "C" __global__ void downsample(
            const float* __restrict__ src,
            float* __restrict__ dst,
            const int batch_size,
            const int src_height,
            const int src_width,
            const int dst_height,
            const int dst_width
        ) {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            const int batch_idx = blockIdx.z;

            if (tx >= dst_width || ty >= dst_height || batch_idx >= batch_size)
                return;

            // 计算缩放比例
            const float scale_x = (float)src_width / dst_width;
            const float scale_y = (float)src_height / dst_height;

            // 计算源图像坐标
            const float src_x = tx * scale_x;
            const float src_y = ty * scale_y;

            // 双线性插值
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            const int x1 = min(x0 + 1, src_width - 1);
            const int y1 = min(y0 + 1, src_height - 1);

            const float wx1 = src_x - x0;
            const float wy1 = src_y - y0;
            const float wx0 = 1.0f - wx1;
            const float wy0 = 1.0f - wy1;

            const int src_offset = batch_idx * src_height * src_width;
            const int dst_offset = batch_idx * dst_height * dst_width;

            // 执行插值
            const float val =
                src[src_offset + y0 * src_width + x0] * wx0 * wy0 +
                src[src_offset + y0 * src_width + x1] * wx1 * wy0 +
                src[src_offset + y1 * src_width + x0] * wx0 * wy1 +
                src[src_offset + y1 * src_width + x1] * wx1 * wy1;

            // 写入结果
            dst[dst_offset + ty * dst_width + tx] = val;
        }
        '''
        return cp.RawKernel(kernel_code, 'downsample')

    def process(self, gpu_images):
        self.downsample_kernel(
            self.grid_size,
            self.block_size,
            (
                gpu_images.ravel(),
                self.d_downsampled,
                self.batch_size,
                self.input_height,
                self.input_width,
                self.output_height,
                self.output_width
            )
        )

        # 返回前景掩码
        return self.d_downsampled




class ViBeGPU:
    def __init__(self, batch_size=10, height=340, width=480, num_samples=20,
                 min_matches=15, radius=5, random_subsample=16):
        """初始化ViBe背景建模器的GPU实现"""
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.min_matches = min_matches
        self.radius = radius
        self.random_subsample = random_subsample

        # CUDA配置
        self.block_size = (16, 16)
        self.grid_size = (
            (width + self.block_size[0] - 1) // self.block_size[0],
            (height + self.block_size[1] - 1) // self.block_size[1],
            batch_size
        )

        # 编译CUDA kernels
        self.init_kernel = self._get_init_kernel()
        self.update_kernel = self._get_update_kernel()

        # 初始化状态
        self.initialized = cp.zeros(batch_size, dtype=cp.bool_)

        # 初始化GPU内存
        self._initialize_gpu_memory()

        # 随机数生成器
        self.rng = cp.random.RandomState()

    def _initialize_gpu_memory(self):
        """初始化GPU内存"""
        # 样本模型 (batch_size, height, width, num_samples)
        self.d_samples = cp.zeros((self.batch_size, self.height, self.width,
                                   self.num_samples), dtype=cp.uint8)

        # 前景掩码
        self.d_foreground_mask = cp.zeros((self.batch_size, self.height, self.width),
                                          dtype=cp.uint8)

        # 随机数缓冲区
        self.d_random = cp.zeros((self.batch_size, self.height, self.width),
                                 dtype=cp.int32)

    def _get_init_kernel(self):
        """初始化kernel"""
        kernel_code = r'''
        extern "C" __global__ void init_vibe(
            unsigned char* __restrict__ samples,
            const float* __restrict__ frame,
            const int height,
            const int width,
            const int num_samples,
            const int batch_size
        ) {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            const int batch_idx = blockIdx.z;

            if (tx >= width || ty >= height || batch_idx >= batch_size)
                return;

            const int pixel_idx = batch_idx * height * width + ty * width + tx;
            const float pixel_value = frame[pixel_idx];
            const int pixel_int = (int)pixel_value;

            // 初始化第一个样本为当前值
            samples[pixel_idx * num_samples] = (unsigned char)pixel_value;

            // 随机选取邻域像素作为其他样本
            const int x_start = max(0, tx - 1);
            const int x_end = min(width - 1, tx + 1);
            const int y_start = max(0, ty - 1);
            const int y_end = min(height - 1, ty + 1);

            #pragma unroll
            for(int n = 1; n < num_samples; n++) {
                // 使用像素值的低位作为简单的随机源
                const int rand_x = x_start + (pixel_int * n) % (x_end - x_start + 1);
                const int rand_y = y_start + (pixel_int * n * n) % (y_end - y_start + 1);
                const int rand_idx = batch_idx * height * width + rand_y * width + rand_x;

                samples[pixel_idx * num_samples + n] = (unsigned char)frame[rand_idx];
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'init_vibe')

    def _get_update_kernel(self):
        """更新kernel"""
        kernel_code = r'''
        extern "C" __global__ void update_vibe(
            unsigned char* __restrict__ samples,
            unsigned char* __restrict__ foreground_mask,
            const float* __restrict__ frame,
            const int* __restrict__ random_nums,
            const int height,
            const int width,
            const int num_samples,
            const int min_matches,
            const int radius,
            const int random_subsample,
            const int batch_size
        ) {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            const int batch_idx = blockIdx.z;

            if (tx >= width || ty >= height || batch_idx >= batch_size)
                return;

            const int pixel_idx = batch_idx * height * width + ty * width + tx;
            const float pixel = frame[pixel_idx];
            const int random_num = random_nums[pixel_idx];

            // 计算与样本的匹配数
            int matches = 0;
            const float radius_squared = radius * radius;

            #pragma unroll
            for(int i = 0; i < num_samples; i++) {
                const float diff = pixel - samples[pixel_idx * num_samples + i];
                if(diff * diff <= radius_squared) {
                    matches++;
                    if(matches >= min_matches)
                        break;
                }
            }

            // 判断是否为前景
            if(matches >= min_matches) {
                foreground_mask[pixel_idx] = 0;  // 背景

                // 随机更新样本集
                if(random_num % random_subsample == 0) {
                    const int rand_idx = (random_num / random_subsample) % num_samples;
                    samples[pixel_idx * num_samples + rand_idx] = (unsigned char)pixel;

                    // 随机更新邻域像素的样本集
                    const int dx = (random_num >> 8) % 3 - 1;  // -1, 0, or 1
                    const int dy = (random_num >> 16) % 3 - 1; // -1, 0, or 1

                    const int nx = tx + dx;
                    const int ny = ty + dy;

                    if(nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const int neighbor_idx = batch_idx * height * width + ny * width + nx;
                        const int rand_sample = (random_num >> 24) % num_samples;
                        samples[neighbor_idx * num_samples + rand_sample] = (unsigned char)pixel;
                    }
                }
            } else {
                foreground_mask[pixel_idx] = 255;  // 前景
            }
        }
        '''
        return cp.RawKernel(kernel_code, 'update_vibe')

    def apply(self, gpu_frames):
        """应用ViBe背景建模

        Args:
            gpu_frames: cupy array (batch_size, height, width)

        Returns:
            cupy array (batch_size, height, width)
        """
        # 生成随机数
        self.d_random = self.rng.randint(0, 0x7fffffff,
                                         size=(self.batch_size, self.height, self.width),
                                         dtype=cp.int32)

        # 处理每个batch
        for batch_idx in range(self.batch_size):
            if not self.initialized[batch_idx]:
                # 初始化阶段
                self.init_kernel(
                    self.grid_size,
                    self.block_size,
                    (
                        self.d_samples.ravel(),
                        gpu_frames.ravel(),
                        self.height,
                        self.width,
                        self.num_samples,
                        self.batch_size
                    )
                )
                self.initialized[batch_idx] = True

            # 更新阶段
            self.update_kernel(
                self.grid_size,
                self.block_size,
                (
                    self.d_samples.ravel(),
                    self.d_foreground_mask.ravel(),
                    gpu_frames.ravel(),
                    self.d_random.ravel(),
                    self.height,
                    self.width,
                    self.num_samples,
                    self.min_matches,
                    self.radius,
                    self.random_subsample,
                    self.batch_size
                )
            )

        return self.d_foreground_mask

    def apply_cpu(self, cpu_frames):
        """CPU接口函数

        Args:
            cpu_frames: numpy array (batch_size, height, width)

        Returns:
            numpy array (batch_size, height, width)
        """
        gpu_frames = cp.asarray(cpu_frames)
        gpu_result = self.apply(gpu_frames)
        return gpu_result.get()




class PostProcessorGPU:
    def __init__(self, history_len=5, num_channels=10, min_count=500, iou_threshold=0.1, border_width=10):
        """初始化后处理器

        Args:
            history_len: 历史帧数量，默认5帧
            num_channels: 分布统计的通道数，默认10
            min_count: 最小像素计数阈值，默认500
            iou_threshold: box合并的IOU阈值，默认0.2
            border_width: 检测框的边界宽度，默认10
        """
        self.history_len = history_len
        self.num_channels = num_channels
        self.min_count = min_count
        self.iou_threshold = iou_threshold
        self.border_width = border_width

        # 初始化历史帧缓存 (5, 10, 340, 480)
        self.mask_history = None

        # CUDA kernels
        self._initialize_kernels()

    def _initialize_kernels(self):
        """初始化CUDA kernels"""
        # Salt and pepper滤波kernel
        self.filter_kernel = cp.RawKernel(r'''
        extern "C" __global__ void salt_pepper_filter(
            const unsigned char* __restrict__ input,
            unsigned char* __restrict__ output,
            const int height,
            const int width
        ) {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;

            if (tx >= width-1 || ty >= height-1 || tx < 1 || ty < 1)
                return;

            float sum = 0.0f;

            // 计算3x3邻域和
            sum += input[(ty-1) * width + (tx-1)];
            sum += input[(ty-1) * width + tx];
            sum += input[(ty-1) * width + (tx+1)];
            sum += input[ty * width + (tx-1)];
            sum += input[ty * width + tx];
            sum += input[ty * width + (tx+1)];
            sum += input[(ty+1) * width + (tx-1)];
            sum += input[(ty+1) * width + tx];
            sum += input[(ty+1) * width + (tx+1)];

            output[ty * width + tx] = (sum > 255.0f * 2.0f) ? 255 : 0;
        }
        ''', 'salt_pepper_filter')

    def process(self, mask_torch):
        """处理掩码图像

        Args:
            mask_torch: CuPy array

        Returns:
            处理后的掩码和检测框列表
        """
        # 初始化历史缓存
        if self.mask_history is None:
            self.mask_history = cp.zeros((self.history_len,) + mask_torch.shape, dtype=cp.uint8)

        # 更新历史缓存
        self.mask_history = cp.roll(self.mask_history, 1, axis=0)
        self.mask_history[0] = mask_torch

        # 计算历史帧的和
        mask_sum = cp.sum(self.mask_history, axis=0)

        # 找出超过阈值的位置并置零
        mask = cp.where(mask_sum > 255, 0, mask_torch)

        # 转换为浮点数并归一化
        mask = mask.astype(cp.float32)
        mask = cp.sum(mask / 10, axis=0)  # 对通道维度求和并归一化

        # 计算分布
        mask_distribution = cp.zeros(self.num_channels)
        for i in range(self.num_channels):
            bin_start = i * 25.5
            bin_end = (i + 1) * 25.5
            mask_temp = cp.where((bin_start < mask) & (mask <= bin_end), 1, 0)
            mask_distribution[i] = cp.sum(mask_temp)

        # 根据分布确定阈值
        idx = cp.where(mask_distribution < self.min_count)[0]
        if len(idx) > 0:
            threshold = idx[0] * 25.5
            mask = cp.where(mask > threshold, 255, 0)

        # 应用salt and pepper滤波
        filtered_mask = cp.zeros_like(mask, dtype=cp.uint8)
        block_size = (16, 16)
        grid_size = (
            (mask.shape[1] + block_size[0] - 1) // block_size[0],
            (mask.shape[0] + block_size[1] - 1) // block_size[1]
        )
        self.filter_kernel(grid_size, block_size, (mask.astype(cp.uint8), filtered_mask,
                                                   mask.shape[0], mask.shape[1]))

        # 转到CPU生成检测框
        cpu_mask = cp.asnumpy(filtered_mask)
        boxes = self._generate_boxes(cpu_mask)
        merged_boxes = self._merge_boxes(boxes) if boxes else []

        return filtered_mask, merged_boxes

    def _generate_boxes(self, mask):
        """生成候选检测框"""
        boxes = []
        y_coords, x_coords = np.where(mask == 255)

        for y, x in zip(y_coords, x_coords):
            box = (
                max(x - self.border_width, 0),
                max(y - self.border_width, 0),
                min(x + self.border_width, mask.shape[1]),
                min(y + self.border_width, mask.shape[0])
            )
            boxes.append(box)

        return boxes

    def _calculate_iou(self, box1, box2):
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union

    def _merge_boxes(self, boxes):
        """合并重叠检测框"""
        if not boxes:
            return []

        # 按面积排序
        sorted_boxes = sorted(boxes,
                              key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                              reverse=True)
        merged_boxes = []

        while len(sorted_boxes) > 0:
            current = sorted_boxes.pop(0)
            to_merge = []

            for box in sorted_boxes[:]:
                if self._calculate_iou(current, box) > self.iou_threshold:
                    to_merge.append(box)
                    sorted_boxes.remove(box)

            if to_merge:
                merged = current
                for box in to_merge:
                    merged = (
                        min(merged[0], box[0]),
                        min(merged[1], box[1]),
                        max(merged[2], box[2]),
                        max(merged[3], box[3])
                    )
                merged_boxes.append(merged)
            else:
                merged_boxes.append(current)

        return merged_boxes









class ImageProcessor:
    def __init__(self):
        self.transformer = PerspectiveTransformer()
        self.downsampler = DownSampler()
        # self.downsampler2 = DownSampler(batch_size=3, input_height=2720, input_width=3840, resize_height=340, resize_width=480)
        self.vibe = ViBeGPU(min_matches=2, radius=10, random_subsample=2)
        self.post_processor = PostProcessorGPU()

    def process(self, cpu_images, cpu_matrices):
        """处理图像

        Args:
            cpu_images: 输入图像，numpy array

        Returns:
            处理后的掩码、RGB图像和检测框
        """
        # 图像变换和下采样
        gpu_images = self.transformer.process(cpu_images, cpu_matrices)
        # [11, 2720, 3840]
        rgb_image = cp.stack([gpu_images[6], gpu_images[4], 0.95 * gpu_images[7]], axis=0)
        rgb_image_new = cp.stack([gpu_images[7], gpu_images[4], gpu_images[6]], axis=-1)
        gpu_images_downsample = self.downsampler.process(gpu_images)
        # [10, 340, 480]
        rgb_image_show = cp.stack([0.95 * gpu_images_downsample[7], gpu_images_downsample[4], gpu_images_downsample[6]], axis=-1)

        # 背景建模
        foreground_masks = self.vibe.apply(gpu_images_downsample)

        # 后处理
        processed_mask, detection_boxes = self.post_processor.process(foreground_masks)

        # 转换结果到CPU
        cpu_mask = cp.asnumpy(processed_mask)
        cpu_rgb = cp.asnumpy(rgb_image)
        cpu_rgb_new = cp.asnumpy(rgb_image_new)
        cpu_rgb_downsample = cp.asnumpy(rgb_image_show.astype(cp.uint8))

        return cpu_mask, cpu_rgb_new, cpu_rgb_downsample, detection_boxes

    def cleanup(self):
        """清理资源"""
        # 清理各个组件
        for attr in ['transformer', 'downsampler', 'vibe', 'post_processor']:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # 清理GPU内存
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()





def LoadMatrix(filepath="/test_images/matrix/"):
    matrices = np.zeros((8, 3, 3))
    for ii in range(8):
        matrix_name = 'M' + str(ii) + '1.npy'
        if ii == 1:
            matrix = np.eye(3)
        else:
            matrix = np.load(filepath + matrix_name)
        matrices[int(ii), :, :] = np.linalg.inv(matrix)
    return matrices





