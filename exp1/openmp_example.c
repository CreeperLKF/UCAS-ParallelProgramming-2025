#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
// 可追加头文件
// 读取和写入的代码可放入指定的进程中进行

#include "immintrin.h"

// #include <assert.h>
// #define KF_STD0
// #define KF_STD1
#define KF_OMP0
// #define KF_OMP1
// #define KF_OMP2

#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

#include <time.h>

static double timestamp;

#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year   = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday   = wtm.wDay;
    tm.tm_hour   = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm. tm_isdst  = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double get_time()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void start_perf()
{
        timestamp=get_time();
}

void end_perf(const char* str)
{
    timestamp=get_time()-timestamp;
    printf("PERF: %s times cost: %.6lfs\n", str, timestamp);
}

int main(int argc, char *argv[]) {
	// 检查是否提供了文件路径参数
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;  // 退出程序，返回错误码
    }

    // 从命令行参数中获取输入文件路径
    const char *input_file = argv[1];
	
    const int rows = 6400;
    const int cols = 6400;

    // 为矩阵分配内存
    int32_t *matrix = (int32_t *)aligned_alloc(1024, rows * cols * sizeof(int32_t));
    int32_t *result_matrix = (int32_t *)aligned_alloc(1024, rows / 2 * cols / 2 * sizeof(int32_t));
    memset(result_matrix, 0, rows / 2 * cols / 2 * sizeof(int32_t));

    // 从二进制文件中读取矩阵数据
    FILE *file = fopen(input_file, "rb");
    if (file != NULL) {
        fread(matrix, sizeof(int32_t), rows * cols, file);
        fclose(file);
        printf("Matrix has been read from random_matrix.bin\n");

    } else {
        printf("Error opening file for reading\n");
    }
    
    int32_t (*matrix_ptr)[cols] = (int32_t (*)[cols])matrix;
    int32_t (*result_matrix_ptr)[cols / 2] = (int32_t (*)[cols / 2])result_matrix;
#ifdef KF_STD0
    int32_t (*inter_matrix_ptr)[cols] = (int32_t (*)[cols])calloc(rows / 2 * cols, sizeof(int32_t));
#endif
    
    start_perf();
	/*******************************************/
	// TODO: 在这里实现openMP或MPI的程序逻辑，以matrix作为输入，以result_matrix作为输出
	/*******************************************/


#ifdef KF_STD0
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            inter_matrix_ptr[i / 800 * 400 + i % 400][j] += matrix_ptr[i][j];
        }
    }

    for (int i = 0; i < rows / 2; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix_ptr[i][j / 400 * 200 + j % 200] += inter_matrix_ptr[i][j];
        }
    }
#endif



#ifdef KF_STD1
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix_ptr[i / 800 * 400 + i % 400][j / 400 * 200 + j % 200] += matrix_ptr[i][j];
        }
    }
#endif

#ifdef KF_OMP0
    #pragma omp parallel for schedule(static) collapse(3) num_threads(8)
    for (int r = 0; r < 6400; r += 800) {
        for (int k = 0; k < 400; k++) {
            for (int j = 0; j < cols; j += 4) {
                __m128i vecA = _mm_load_si128((__m128i*)&matrix_ptr[r + k][j]);
                __m128i vecB = _mm_load_si128((__m128i*)&matrix_ptr[r + k + 400][j]);
                vecA = _mm_add_epi32(vecA, vecB);
                _mm_store_si128((__m128i*)&matrix_ptr[r + k][j], vecA);
                // matrix_ptr[k][j] += matrix_ptr[k + 400][j];
                // matrix_ptr[k + 800][j] += matrix_ptr[k + 1200][j];
                // matrix_ptr[k + 1600][j] += matrix_ptr[k + 2000][j];
                // matrix_ptr[k + 2400][j] += matrix_ptr[k + 2800][j];
                // matrix_ptr[k + 3200][j] += matrix_ptr[k + 3600][j];
                // matrix_ptr[k + 4000][j] += matrix_ptr[k + 4400][j];
                // matrix_ptr[k + 4800][j] += matrix_ptr[k + 5200][j];
                // matrix_ptr[k + 5600][j] += matrix_ptr[k + 6000][j];
            }
        }
    }
    #pragma omp parallel for schedule(static) collapse(4) num_threads(16)
    for (int i = 0; i < rows / 2; i += 400) {
        for (int ik = 0; ik < 400; ik ++) {
            for (int j = 0; j < cols / 2; j += 1600) {
                for (int r = 0; r < 1600; r += 200) {
                    for (int k = 0; k < 200; k += 4) {
                        __m128i vecA = _mm_load_si128((__m128i*)&matrix_ptr[(i << 1) + ik][(j << 1) + (r << 1) + k]);
                        __m128i vecB = _mm_load_si128((__m128i*)&matrix_ptr[(i << 1) + ik][(j << 1) + (r << 1) + k + 200]);
                        __m128i vecC = _mm_add_epi32(vecA, vecB);
                        _mm_store_si128((__m128i*)&result_matrix_ptr[i + ik][j + k + r], vecC);
                        // result_matrix_ptr[i + ik][j + k] = matrix_ptr[(i << 1) + ik][(j << 1) + k] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 200];
                        // result_matrix_ptr[i + ik][j + k + 200] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 400] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 600];
                        // result_matrix_ptr[i + ik][j + k + 400] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 800] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 1000];
                        // result_matrix_ptr[i + ik][j + k + 600] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 1200] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 1400];
                        // result_matrix_ptr[i + ik][j + k + 800] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 1600] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 1800];
                        // result_matrix_ptr[i + ik][j + k + 1000] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 2000] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 2200];
                        // result_matrix_ptr[i + ik][j + k + 1200] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 2400] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 2600];
                        // result_matrix_ptr[i + ik][j + k + 1400] = matrix_ptr[(i << 1) + ik][(j << 1) + k + 2800] + matrix_ptr[(i << 1) + ik][(j << 1) + k + 3000];
                    }
                }
            }
        }
    }
#endif

#ifdef KF_OMP1
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int (j << 1) = 0; j < cols; j++) {
            int result_i = i / 800 * 400 + i % 400;
            int result_j = j / 400 * 200 + j % 200;
            #pragma omp atomic
            result_matrix_ptr[result_i][result_j] += matrix_ptr[i][j];
        }
    }
#endif

#ifdef KF_OMP2
    // best performance with 0.02s
    const int half_rows = rows / 2;
    const int half_cols = cols / 2;
    #pragma omp parallel for collapse(4) schedule(static)
    for (int x = 0; x < 2; x ++) {
        for (int y = 0; y < 2; y ++) {
            // #pragma omp parallel for collapse(2) schedule(static)
            for (int ib = 0; ib < half_rows; ib += 400) {
                for (int jb = 0; jb < half_cols; jb += 200) {
                    int blockX = ib / 400 * 800 + 400 * x;
                    int blockY = jb / 200 * 400 + 200 * y;
                    for (int i = 0; i < 400; i++) {
                        for (int j = 0; j < 200; j++) {
                            int result_i = ib + i;
                            int result_j = jb + j;
                            int target_i = blockX + i;
                            int target_j = blockY + j;
                            #pragma omp atomic
                            result_matrix_ptr[result_i][result_j] += matrix_ptr[target_i][target_j];
                        }
                    }
                }
            }
        }
    }
#endif

    end_perf("openMP or MPI");

    // 将矩阵写入二进制文件
    file = fopen("result_matrix.bin", "wb");
    if (file != NULL) {
        fwrite(result_matrix, sizeof(int32_t), rows / 2 * cols / 2, file);
        fclose(file);
        printf("Matrix has been written to result_matrix.bin\n");
    } else {
        printf("Error opening file for writing\n");
    }

    // 释放内存
    free(matrix);
    free(result_matrix);

#ifdef KF_STD0
    free(inter_matrix_ptr);
#endif

    return 0;
}