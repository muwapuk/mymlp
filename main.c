#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define GRAPHICS_API_OPENGL_43
#include "glad.h"
#include "raylib.h"
#include "rlgl.h"

#define WINDOW_SIZE_W 1920
#define WINDOW_SIZE_H 1080

#define COMPUTE_SHADER_PATH "./mat_dot.glsl"
#define NN_FORWARD_SHADER_PATH "./nn_forward.comp"

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
#define da_append(da, item)\
    do {\
        if((da).count >= (da).capacity) {\
            if((da).capacity == 0) (da).capacity = 8;\
            else (da).capacity*=2;\
            (da).items = realloc((da).items, (da).capacity*sizeof(*(da).items));\
        }\
        (da).items[(da).count++] = item;\
    } while(0)
#define da_at(da, i)(da).items[i]

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Mat;
typedef struct {
    size_t cols;
    float *data;
} Row;

#define MAT_AT(m, row, col) (m).data[(row)*(m).cols + (col)]
Mat mat_alloc(size_t rows, size_t cols);
Row mat_row(Mat m, size_t row);
void mat_free(Mat m);
void mat_rand(Mat m, float min, float max);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_activ_func(Mat m);
void mat_print(Mat m, size_t padding);
void mat_zero(Mat m);

#define ROW_AT(r, col) (r).data[col]
Mat row_as_mat(Row row);
#define row_alloc(cols) mat_row(mat_alloc(1, cols), 0)
#define row_free(r) mat_free(row_as_mat(r))
Row row_slice(Row row, size_t i, size_t cols); // TODO
#define row_rand(r, min, max) mat_rand(row_as_mat(r), min, max)
#define row_fill(r, x) mat_fill(row_as_mat(r), x);
#define row_print(r, padding) mat_print(row_as_mat(r), padding)
#define row_copy(dst, src) mat_copy(row_as_mat(dst), row_as_mat(src))
#define row_zero(r) mat_zero(row_as_mat(r))

float rand_float(void);

Mat row_as_mat(Row row);

typedef struct {
    size_t layers_count;
    size_t *layers_sizes;

    Row *ns;   // Neurons values
    Mat *ws;  // Weights 
    Row *bs;   // Biases
} Network;
#define NN_INPUT(nn) (assert((nn).layers_count > 0), (nn).ns[0])
#define NN_OUTPUT(nn) (assert((nn).layers_count > 0), (nn).ns[(nn).layers_count-1])

float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}
static inline float activation_func(float x) {
    return sigmoid(x);
}
// nn_layout for example {2, 5, 5, 2}
Network nn_alloc(size_t nn_layout[], size_t nn_layout_size);
void nn_free(Network nn);
void nn_rand_weights_biases(Network nn, float min, float max);
void nn_forward(Network nn, Row input_data, bool *should_stop);
void nn_backpropagation(Network nn,
                     Row target_output, 
                     Mat *dw,
                     Row *db,
                     bool *should_stop);
void train(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop);
void nn_forward_gpu(Network nn, Row input_data, bool *should_stop);
void nn_backpropagation_gpu(Network nn,
                     Row target_output, 
                     Mat *dw,
                     Row *db,
                     bool *should_stop);
void train_gpu(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop);
float loss(Row output, Row target);


typedef struct {
    Vector2 pos;
    Color color;
} Point;

typedef struct {
    Point *items;  
    size_t count;
    size_t capacity;
} Points;

void *nn_worker(void *should_stop);
void nn_worker_train();
void nn_worker_gen_texture();
void restart_nn_worker();

bool nn_update_request = false;
bool nn_texture_rdy = false;

Network nn;
size_t layout[] = { 2, 161, 16, 1 };

Points points = {0};
float rad = 5;
int rect_size = 5;

Color *pixel_front_buf;
Color *pixel_back_buf;

unsigned int nn_train_program;

void nn_retrain();
void nn_gen_texture();

int main()
{
    srand(42);
    InitWindow(WINDOW_SIZE_W, WINDOW_SIZE_H, "Window");
    //SetTargetFPS(144);

    bool worker_should_stop = false; 

#define SHOWLOSS

    pixel_front_buf = malloc((size_t)WINDOW_SIZE_W * WINDOW_SIZE_H * sizeof(Color));
    pixel_back_buf = malloc((size_t)WINDOW_SIZE_W * WINDOW_SIZE_H * sizeof(Color));
    for(int i = 0; i < WINDOW_SIZE_W * WINDOW_SIZE_H; i++) pixel_front_buf[i] = DARKGRAY;
    Image nn_image = {
        .data = pixel_front_buf,
        .width = WINDOW_SIZE_W,
        .height = WINDOW_SIZE_H,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    Texture nn_texture = LoadTextureFromImage(nn_image);
    ImageClearBackground(&nn_image, DARKGRAY);

    nn = nn_alloc(layout, ARRAY_LEN(layout));

    char *nn_train_shader_src = LoadFileText(NN_FORWARD_SHADER_PATH);
    unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    UnloadFileText(nn_train_shader_src);
    nn_train_program = rlLoadShaderProgramCompute(nn_train_shader);
    
    while(!WindowShouldClose()) {
        if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            Point p = (Point){ .pos.x = GetMousePosition().x, .pos.y = GetMousePosition().y, GREEN };    
            da_append(points, p);
            nn_retrain();
            nn_gen_texture();
        } else if(IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            Point p = (Point){ .pos.x = GetMousePosition().x, .pos.y = GetMousePosition().y, BLUE };    
            da_append(points, p);
            nn_retrain();
            nn_gen_texture();
        }
        if(nn_texture_rdy) {
            UpdateTexture(nn_texture, pixel_front_buf); 
            nn_texture_rdy = false;
        }
        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(nn_texture, 0, 0, WHITE);
            for(size_t i = 0; i < points.count; i++) {
                DrawCircle(da_at(points, i).pos.x, da_at(points, i).pos.y, rad, da_at(points, i).color);
                DrawCircleLines(da_at(points, i).pos.x, da_at(points, i).pos.y, rad + .5f, BLACK);
            }
            DrawFPS(10, 10);
        EndDrawing();
    }
    worker_should_stop = true;
    nn_update_request = true;


    UnloadTexture(nn_texture);
    rlUnloadShaderProgram(nn_train_program);
    free(pixel_front_buf);
    nn_free(nn);
    CloseWindow();

    return 0;
}    
void nn_retrain() 
{
    Mat input = mat_alloc(points.count, 2);
    Mat target = mat_alloc(points.count, 1);
    for(size_t i = 0; i < points.count; i++) {
        MAT_AT(input, i, 0) = da_at(points, i).pos.x/WINDOW_SIZE_W;
        MAT_AT(input, i, 1) = da_at(points, i).pos.y/WINDOW_SIZE_H;
        MAT_AT(target, i, 0) = ColorToInt(da_at(points, i).color) == ColorToInt(GREEN) ? 0 : 1;
    }
    train(nn, 1e-1f, input, target, (size_t)1e4, &nn_update_request);

    mat_free(input);
    mat_free(target);
}
void nn_gen_texture() 
{
    Row input = row_alloc(2);
    for(int height = 0; height < WINDOW_SIZE_H; height++) {
        for(int width = 0; width < WINDOW_SIZE_W; width++) {
            int index = height*WINDOW_SIZE_W + width;
            ROW_AT(input, 0) = (float)width/WINDOW_SIZE_W;
            ROW_AT(input, 1) = (float)height/WINDOW_SIZE_H;
            nn_forward(nn, input, &nn_update_request);
            if(nn_update_request) {
                row_free(input);
                return;
            }
            unsigned char green_value = (unsigned char)(255*ROW_AT(NN_OUTPUT(nn), 0));
            Color pix = {
                0,
                255-green_value,
                green_value,
                255,
            };
            pixel_back_buf[index] = pix; 
        }

    }
    Color *tmp_buf = pixel_front_buf;
    pixel_front_buf = pixel_back_buf;
    pixel_back_buf = tmp_buf;
    nn_texture_rdy = true;
    row_free(input);
}
inline float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}
Mat mat_alloc(size_t rows, size_t cols) {
    assert(rows != 0 && cols != 0);
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.data = calloc(rows*cols, sizeof(*m.data));
    assert(m.data != NULL);

    return m;
}
Row mat_row(Mat m, size_t row)
{
    return (Row) { 
        .cols = m.cols,
        .data = &MAT_AT(m, row, 0)
    };
}
void mat_free(Mat mat)
{
    if(mat.data != NULL) {
        free(mat.data);
    }
}
void mat_rand(Mat m, float min, float max)
{
    for(size_t row = 0; row < m.rows; row++) {
        for(size_t col = 0; col < m.cols; col++) {
            MAT_AT(m, row, col) = rand_float()*(max - min) + min;
        }
    }
}
void mat_dot(Mat dst, Mat a, Mat b)
{
    if(a.cols !=  b.rows 
    || dst.rows !=  a.rows
    || dst.cols !=  b.cols) {
        fprintf(stderr, "Failed to get dot prodact of matrices. Bad dimensions!\n");
        return;
    }
    size_t additions = a.cols;
    for(size_t row = 0; row < dst.rows; row++) {
        for(size_t col = 0; col < dst.cols; col++) {
            MAT_AT(dst, row, col) = 0;
            for(size_t add_num = 0; add_num < additions; add_num++) {
                MAT_AT(dst, row, col) += MAT_AT(a, row, add_num)*MAT_AT(b, add_num, col);
            }
        }
    }
}
void mat_sum(Mat dst, Mat a)
{
    if(dst.rows != a.rows || dst.cols != a.cols) {
        fprintf(stderr, "Failed to sum matrices. Rows and colums are not the same!\n");
        return;
    }
    for(size_t row = 0; row < dst.rows; row++) { 
        for(size_t col = 0; col < dst.cols; col++) {
            MAT_AT(dst, row, col) += MAT_AT(a, row, col);
        }
    }
}
void mat_activ_func(Mat m) 
{
    for(size_t row = 0; row < m.rows; row++) {
        for(size_t col = 0; col < m.cols; col++) {
            MAT_AT(m, row, col) = activation_func(MAT_AT(m, row, col));
        }
    }
}
void mat_print(Mat m, size_t padding)
{
    printf("%*s = [\n", (int) padding, "");
    for (size_t row = 0; row < m.rows; ++row) {
        printf("%*s    ", (int) padding, "");
        for (size_t col = 0; col < m.cols; ++col) {
            printf("%f ", MAT_AT(m, row, col));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}
void mat_zero(Mat m)
{
    memset(m.data, 0, sizeof(*m.data)*m.rows*m.cols);
}
Mat row_as_mat(Row row)
{
    return (Mat) {
        .rows = 1,
        .cols = row.cols,
        .data = row.data,
    };
}
Network nn_alloc(size_t *nn_layout, size_t nn_layout_size) 
{ 
    Network nn;
    nn.layers_count = nn_layout_size;
    nn.layers_sizes = calloc(nn_layout_size, sizeof(*nn.layers_sizes));
    assert(nn.layers_sizes != NULL);
    // Set layers sizes
    for(size_t l = 0; l < nn_layout_size; l++) {
        nn.layers_sizes[l] = nn_layout[l]; 
    }
    // Allocate neurons, weights and biases
    nn.ns = malloc(sizeof(*nn.ns)*nn_layout_size);     //
    nn.ws = malloc(sizeof(*nn.ws)*(nn_layout_size-1));   // -1 is because nn_layout_size includes inputs
    nn.bs = malloc(sizeof(*nn.bs)*(nn_layout_size-1));   //
    for(size_t l = 0; l < nn_layout_size-1; l++) {
        nn.ns[l] = row_alloc(nn_layout[l]);
        nn.ws[l] = mat_alloc(nn_layout[l], nn_layout[l+1]);
        nn.bs[l] = row_alloc(nn_layout[l+1]);
    }    
    nn_rand_weights_biases(nn, -1.0f, 1.0f);
    nn.ns[nn_layout_size-1] = row_alloc(nn_layout[nn_layout_size-1]);
    return nn;
}
void nn_free(Network nn) 
{
    assert(nn.layers_sizes != NULL);
    for(size_t l = 0; l < nn.layers_count; l++) {
        row_free(nn.ns[l]);

        if(l < nn.layers_count-1) {
            mat_free(nn.ws[l]);
            row_free(nn.bs[l]);
        }
    }
    free(nn.ns);
    free(nn.ws);
    free(nn.bs);
    free(nn.layers_sizes);
}
void nn_rand_weights_biases(Network nn, float min, float max)
{
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        mat_rand(nn.ws[l], min, max);
        row_rand(nn.bs[l], min, max);
    }    
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than weights and biases arrays by 1 
// thus the weights, biases and deltas look like they are from previous layer
void nn_forward(Network nn, Row input_data, bool *should_stop)
{
    if(!nn.layers_sizes 
    || !nn.ns
    || !nn.ws
    || !nn.bs) {
        fprintf(stderr, "Invalid nn\n");
        return;
    }
    if(!input_data.data) {
        fprintf(stderr, "Invalid nn forward data\n");
        return;
    }
    if(*should_stop) return;
    for(size_t data_num = 0; data_num < input_data.cols; data_num++) {
        ROW_AT(nn.ns[0], data_num) = ROW_AT(input_data, data_num);
    }
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        if(*should_stop) return;
        mat_dot(row_as_mat(nn.ns[l+1]), row_as_mat(nn.ns[l]), nn.ws[l]);
        mat_sum(row_as_mat(nn.ns[l+1]), row_as_mat(nn.bs[l]));
        mat_activ_func(row_as_mat(nn.ns[l+1]));
    }
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than deltas, weights and biases arrays by 1
// thus the weights, biases and deltas look like they are from previous layer
void nn_backpropagation(Network nn,
                     Row target_output, 
                     Mat *dW,
                     Row *dB,
                     bool *should_stop) {
    if(!nn.layers_sizes) {
        fprintf(stderr, "Invalid nn\n");
        return;
    }
    if(target_output.cols != nn.layers_sizes[nn.layers_count-1]) {
        fprintf(stderr, "Bad target data dimensions!\n");
        return;
    }
    if(*should_stop) return;
    Row *delta = malloc(sizeof(*delta)*(nn.layers_count-1));
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        // l+1 cuz delta size based on neurons layer size
        delta[l] = row_alloc(nn.layers_sizes[l+1]);
    }
    for(size_t col = 0; col < target_output.cols; col++) {
        float A = (ROW_AT(nn.ns[last_layer], col));
        ROW_AT(delta[last_layer-1], col)   // delta[l] = (Al - y) * Al * (1 - Al)
            = (A
            - ROW_AT(target_output, col))
            * A * (1 - A);
    } 
    for(ssize_t l = (ssize_t)last_layer-1; l >= 0; l--) { 
        size_t current_layer = nn.layers_sizes[l];
        size_t next_layer = nn.layers_sizes[l+1];
        // Weights
        for(size_t row = 0; row < current_layer; row++) {
            if(*should_stop) goto nn_backpropagation_exit;
            for(size_t col = 0; col < next_layer; col++) {
                MAT_AT(dW[l], row, col) // delta Wl = delta[l] * Al-1
                    = ROW_AT(delta[l], col) 
                    * ROW_AT(nn.ns[l], row);
            }
        }
        // Biases
        for(size_t col = 0; col < next_layer; col++) {
            ROW_AT(dB[l], col) // delta bl = delta[l]
                = ROW_AT(delta[l], col);
        }
        // Deltas for prev layer if not last layer
        if(l > 0) {
            // delta[l]-1 = delta[l] * Wl * Al-1 * (1 - Al-1)
            for(size_t row = 0; row < current_layer; row++) {
                if(*should_stop) goto nn_backpropagation_exit;
                float weightDeltaSum = 0.f;
                for(size_t col = 0; col < next_layer; col++) {
                    weightDeltaSum 
                        += ROW_AT(delta[l], col) * MAT_AT(nn.ws[l], row, col);
                }
                float Aprev = ROW_AT(nn.ns[l], row);
                ROW_AT(delta[l-1], row) 
                    = weightDeltaSum * Aprev * (1.f - Aprev);
            }
        }
    }
nn_backpropagation_exit:
    for(size_t l = 0; l < last_layer; l++)
        row_free(delta[l]);
    free(delta);
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than deltas, weights and biases arrays by 1
// thus the weights, biases and deltas look like they are from previous layer
void train(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop)
{
    Mat *dWsum = malloc(sizeof(*dWsum)*(nn.layers_count-1));
    Row *dBsum = malloc(sizeof(*dBsum)*(nn.layers_count-1));

    Mat *dW = malloc(sizeof(*dW)*(nn.layers_count-1));
    Row *dB = malloc(sizeof(*dB)*(nn.layers_count-1));
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        dW[l] = mat_alloc(nn.layers_sizes[l], nn.layers_sizes[l+1]);
        dB[l] = row_alloc(nn.layers_sizes[l+1]);

        dWsum[l] = mat_alloc(nn.layers_sizes[l], nn.layers_sizes[l+1]);
        dBsum[l] = row_alloc(nn.layers_sizes[l+1]);
    }
#ifdef SHOWLOSS
    size_t lossFreq = (epoch_count+9) / 10;
#endif
    for(size_t epoch = 0; epoch < epoch_count; epoch++) {
#ifdef SHOWLOSS
        float lossVal = 0;
#endif
        size_t data_size = input_data.rows;
        for(size_t data_num = 0; data_num < data_size; data_num++) {
            nn_forward_gpu(nn, mat_row(input_data, data_num), should_stop);
            nn_backpropagation(nn,
                            mat_row(target_output, data_num),
                            dW,
                            dB,
                            should_stop); 
            for(size_t l = 0; l < last_layer; l++) {
                if(*should_stop) goto nn_train_exit;
                mat_sum(dWsum[l], dW[l]);
                mat_sum(row_as_mat(dBsum[l]), row_as_mat(dB[l]));
            }
#ifdef SHOWLOSS
            lossVal += loss(NN_OUTPUT(nn), mat_row(target_output, data_num));
#endif
        }
#ifdef SHOWLOSS
        lossVal /= (float)data_size;
        if(epoch % lossFreq == 0) {
            printf("%f\n", lossVal);
        }
#endif
        // Adjust weights and biases
        for(size_t l = 0; l < last_layer; l++) {
            size_t current_layer = nn.layers_sizes[l];
            size_t next_layer = nn.layers_sizes[l+1];
            for(size_t row = 0; row < current_layer; row++) {
                if(*should_stop) goto nn_train_exit;
                for(size_t col = 0; col < next_layer; col++) {
                    MAT_AT(nn.ws[l], row, col)
                        -= learning_rate * MAT_AT(dWsum[l], row, col)/(float)data_size;
                }
            }
            for(size_t col = 0; col < next_layer; col++) {
                MAT_AT(nn.bs[l], 0, col)
                    -= learning_rate * MAT_AT(dBsum[l], 0, col)/(float)data_size;
            }
            mat_zero(dWsum[l]);
            row_zero(dBsum[l]);
        }
    }
nn_train_exit:
    for(size_t l = 0; l < last_layer; l++) {
        mat_free(dW[l]);
        row_free(dB[l]);
        mat_free(dWsum[l]);
        row_free(dBsum[l]);
    }

    free(dW);
    free(dB);
    free(dWsum);
    free(dBsum);
}
void nn_forward_gpu(Network nn, Row input_data, bool *should_stop)
{
    if(!nn.layers_sizes 
    || !nn.ns
    || !nn.ws
    || !nn.bs) {
        fprintf(stderr, "Invalid nn\n");
        return;
    }
    if(!input_data.data) {
        fprintf(stderr, "Invalid nn forward data\n");
        return;
    }
    //char *nn_train_shader_src = LoadFileText(NN_FORWARD_SHADER_PATH);
    //unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    //UnloadFileText(nn_train_shader_src);
    //unsigned int nn_train_program = rlLoadShaderProgramCompute(nn_train_shader);

    // TODO: if(*should_stop) return;
    size_t n_elements_count = 0;
    size_t w_elements_count = 0;
    size_t b_elements_count = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        n_elements_count = nn.layers_sizes[l] > n_elements_count ? nn.layers_sizes[l] : n_elements_count;
        w_elements_count += nn.ws[l].cols*nn.ws[l].rows;
        b_elements_count += nn.bs[l].cols;
    }
    n_elements_count = 
        nn.layers_sizes[nn.layers_count-1] > n_elements_count 
        ? nn.layers_sizes[nn.layers_count-1] 
        : n_elements_count;

    float *n_data = calloc(n_elements_count, sizeof(*n_data));
    float *w_data = malloc(w_elements_count*sizeof(*w_data));
    float *b_data = malloc(b_elements_count*sizeof(*b_data));
    
    size_t w_pos = 0;
    size_t b_pos = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        memcpy(w_data+w_pos, nn.ws[l].data, nn.ws[l].cols*nn.ws[l].rows*sizeof(*w_data));
        memcpy(b_data+b_pos, nn.bs[l].data, nn.bs[l].cols*sizeof(*b_data));

        b_pos += nn.bs[l].cols;
        w_pos += nn.ws[l].cols*nn.ws[l].rows;
    }
    memcpy(n_data, input_data.data, input_data.cols*sizeof(*n_data));

    unsigned int ssbo_input_neurons = rlLoadShaderBuffer(n_elements_count*sizeof(*n_data), n_data, RL_STREAM_COPY);
    unsigned int ssbo_output_neurons = rlLoadShaderBuffer(n_elements_count*sizeof(*n_data), 0, RL_STREAM_COPY);
    unsigned int ssbo_weights = rlLoadShaderBuffer(w_elements_count*sizeof(*w_data), w_data, RL_STATIC_DRAW);
    unsigned int ssbo_biases = rlLoadShaderBuffer(b_elements_count*sizeof(*b_data), b_data, RL_STATIC_DRAW);

    free(n_data);
    free(w_data);
    free(b_data);

    int u_input_size = rlGetLocationUniform(nn_train_program, "u_input_size");
    int u_output_size = rlGetLocationUniform(nn_train_program, "u_output_size");
    int u_weights_offset = rlGetLocationUniform(nn_train_program, "u_weights_offset");
    int u_biases_offset = rlGetLocationUniform(nn_train_program, "u_biases_offset");

    size_t weights_offset = 0;
    size_t biases_offset = 0;

    rlEnableShader(nn_train_program);
    rlBindShaderBuffer(ssbo_weights, 2);
    rlBindShaderBuffer(ssbo_biases, 3);

    size_t workgroup_size = 128;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlBindShaderBuffer(ssbo_input_neurons, 0);
        rlBindShaderBuffer(ssbo_output_neurons, 1);

        rlSetUniform(u_input_size, &nn.layers_sizes[l], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_output_size, &nn.layers_sizes[l+1], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_weights_offset, &weights_offset, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_biases_offset, &biases_offset, RL_SHADER_UNIFORM_UINT, 1);
        
        size_t workgroups_num = (nn.layers_sizes[l+1]+workgroup_size-1)/workgroup_size;
        rlComputeShaderDispatch(workgroups_num, 1, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        rlReadShaderBuffer(ssbo_output_neurons, nn.ns[l+1].data, nn.ns[l+1].cols*sizeof(*nn.ns->data), 0);

        // Swap input and output neuron buffers for next layer
        unsigned int tmp = ssbo_input_neurons;
        ssbo_input_neurons = ssbo_output_neurons;
        ssbo_output_neurons = tmp;

        weights_offset += nn.ws[l].cols*nn.ws[l].rows;
        biases_offset += nn.bs[l].cols;
    }

    rlDisableShader();

    rlUnloadShaderBuffer(ssbo_input_neurons);
    rlUnloadShaderBuffer(ssbo_output_neurons);
    rlUnloadShaderBuffer(ssbo_weights);
    rlUnloadShaderBuffer(ssbo_biases);
} 
float loss(Row output, Row target)
{
    float loss = 0.f;
    for(size_t col = 0; col < output.cols; col++) {
        float diff = ROW_AT(output, col) - ROW_AT(target, col);
        loss += 0.5f * diff * diff;
    }
    return loss;
}
