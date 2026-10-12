#include "nn_gpu.h"
#include "raylib.h"
#include "rlgl.h"
#include "glad.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

unsigned int nn_shader_program;
bool is_weights_biases_init = false;

unsigned *activations_ssbos;
unsigned *weights_ssbos;
unsigned *biases_ssbos;
unsigned target_ssbo;
unsigned *gradients_ssbos;
unsigned *w_gradients_ssbos;
unsigned *b_gradients_ssbos;
unsigned *w_gradients_total_ssbos;
unsigned *b_gradients_total_ssbos;

int u_batch_size;
int u_input_size;
int u_output_size;
int u_is_gradients_increase;
int u_is_gradients_apply;
int u_learning_rate;

int u_is_backprop;
int u_calculate_loss;

static void init_weights_and_biases_buffers(Network nn) 
{
    weights_ssbos = malloc((nn.layers_count-1)*sizeof(*activations_ssbos));
    biases_ssbos = malloc((nn.layers_count-1)*sizeof(*activations_ssbos));
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        weights_ssbos[l] = rlLoadShaderBuffer(
            nn.ws[l].cols*nn.ws[l].rows*sizeof(*nn.ws->data),
            nn.ws[l].data,
            RL_DYNAMIC_COPY);
        biases_ssbos[l] = rlLoadShaderBuffer(
            nn.bs[l].cols*sizeof(*nn.bs->data),
            nn.bs[l].data,
            RL_DYNAMIC_COPY);
    }
}
void init_nn_gpu(Network nn)
{
    // Init shader
    char *nn_train_shader_src = LoadFileText(NN_TRAIN_SHADER_SRC);
    unsigned nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    UnloadFileText(nn_train_shader_src);
    nn_shader_program = rlLoadShaderProgramCompute(nn_train_shader);
    rlEnableShader(nn_shader_program);

    free(activations_ssbos);
    free(weights_ssbos);
    free(biases_ssbos);

    init_weights_and_biases_buffers(nn);
    is_weights_biases_init = true;

    u_batch_size = rlGetLocationUniform(nn_shader_program, "u_batch_size");
    u_input_size = rlGetLocationUniform(nn_shader_program, "u_input_size");
    u_output_size = rlGetLocationUniform(nn_shader_program, "u_output_size");
    u_is_backprop = rlGetLocationUniform(nn_shader_program, "u_is_backprop");
    u_calculate_loss = rlGetLocationUniform(nn_shader_program, "u_calculate_loss");
    u_is_gradients_increase = rlGetLocationUniform(nn_shader_program, "u_is_gradients_increase");
    u_is_gradients_apply = rlGetLocationUniform(nn_shader_program, "u_is_gradients_apply");
    u_learning_rate = rlGetLocationUniform(nn_shader_program, "u_learning_rate");
}
static void free_weights_and_biases_buffers(Network nn)
{
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlUnloadShaderBuffer(weights_ssbos[l]);
        rlUnloadShaderBuffer(biases_ssbos[l]);
    }
    free(weights_ssbos);
    free(biases_ssbos);
}
void reset_nn_gpu(Network nn)
{
    //TODO: Unload buffers
    free_weights_and_biases_buffers(nn);

    rlUnloadShaderProgram(nn_shader_program);
}
static void init_activation_buffers(Network nn, Mat input)
{
    activations_ssbos = malloc((nn.layers_count)*sizeof(*activations_ssbos));
    size_t batch_size = input.rows;
    activations_ssbos[0] = rlLoadShaderBuffer(
        input.cols*batch_size*sizeof(*input.data),
        input.data,
        RL_DYNAMIC_COPY);
    for(size_t l = 1; l < nn.layers_count; l++) {
        activations_ssbos[l] = rlLoadShaderBuffer(
            nn.ns[l].cols*batch_size*sizeof(*input.data),
            NULL,
            RL_DYNAMIC_COPY);
    }
}
static void free_activation_buffers(Network nn)
{
    for(size_t l = 0; l < nn.layers_count; l++) {
        rlUnloadShaderBuffer(activations_ssbos[l]);
    }
    free(activations_ssbos); 
}
void nn_forward_gpu(Network nn, Mat input_data, Mat output_data, bool *should_stop)
{
    if(!is_weights_biases_init) return;
    init_activation_buffers(nn, input_data);
    unsigned batch_size = input_data.rows;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        unsigned in_size = nn.layers_sizes[l];
        unsigned out_size = nn.layers_sizes[l + 1];
        // Bind buffers
        rlBindShaderBuffer(activations_ssbos[l], 0);
        rlBindShaderBuffer(activations_ssbos[l+1], 1);
        rlBindShaderBuffer(weights_ssbos[l], 2);
        rlBindShaderBuffer(biases_ssbos[l], 3);
        // Set variables
        rlSetUniform(u_is_backprop, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
        rlSetUniform(u_is_gradients_increase, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
        rlSetUniform(u_is_gradients_apply, &(int){0}, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_input_size, &in_size, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_output_size, &out_size, RL_SHADER_UNIFORM_UINT, 1);
        // Dispatch
        size_t total_threads = batch_size * nn.layers_sizes[l + 1];
        size_t groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        rlComputeShaderDispatch(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    rlReadShaderBuffer(
        activations_ssbos[nn.layers_count-1],
        output_data.data,
        output_data.rows*output_data.cols*sizeof(*output_data.data),
        0);
    free_activation_buffers(nn);
} 
static void init_gradients_buffers(Network nn, size_t batch_size)
{
    gradients_ssbos = malloc((nn.layers_count-1)*sizeof(*gradients_ssbos));
    w_gradients_ssbos = malloc((nn.layers_count-1)*sizeof(*w_gradients_ssbos));
    b_gradients_ssbos = malloc((nn.layers_count-1)*sizeof(*b_gradients_ssbos));
    w_gradients_total_ssbos = malloc((nn.layers_count-1)*sizeof(*w_gradients_total_ssbos));
    b_gradients_total_ssbos = malloc((nn.layers_count-1)*sizeof(*b_gradients_total_ssbos));
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        gradients_ssbos[l] = rlLoadShaderBuffer(
            nn.ns[l+1].cols*batch_size*sizeof(*nn.ns->data),
            NULL,
            RL_DYNAMIC_COPY);
        w_gradients_ssbos[l] = rlLoadShaderBuffer(
            nn.ws[l].rows*nn.ws[l].cols*batch_size*sizeof(*nn.ws->data),
            NULL,
            RL_DYNAMIC_COPY);
        b_gradients_ssbos[l] = rlLoadShaderBuffer(
            nn.bs[l].cols*batch_size*sizeof(*nn.bs->data),
            NULL,
            RL_DYNAMIC_COPY);
        w_gradients_total_ssbos[l] = rlLoadShaderBuffer(
            nn.ws[l].rows*nn.ws[l].cols*sizeof(*nn.ws->data),
            NULL,
            RL_DYNAMIC_COPY);
        b_gradients_total_ssbos[l] = rlLoadShaderBuffer(
            nn.bs[l].cols*sizeof(*nn.bs->data),
            NULL,
            RL_DYNAMIC_COPY);
    }

}
static void free_gradients_buffers(Network nn)
{
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlUnloadShaderBuffer(gradients_ssbos[l]);
        rlUnloadShaderBuffer(w_gradients_ssbos[l]);
        rlUnloadShaderBuffer(b_gradients_ssbos[l]);
        rlUnloadShaderBuffer(w_gradients_total_ssbos[l]);
        rlUnloadShaderBuffer(b_gradients_total_ssbos[l]);
    }
    free(gradients_ssbos); 
    free(w_gradients_ssbos); 
    free(b_gradients_ssbos); 
    free(w_gradients_total_ssbos); 
    free(b_gradients_total_ssbos); 
}
static void init_target_buffer(Mat target_output)
{
    target_ssbo = rlLoadShaderBuffer(
        target_output.rows*target_output.cols*sizeof(*target_output.data),
        target_output.data,
        RL_DYNAMIC_COPY);
}
static void free_target_buffer() 
{
    rlUnloadShaderBuffer(target_ssbo);
} 
void nn_train_gpu(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop)
{
    if(!is_weights_biases_init) return;

    unsigned batch_size = input_data.rows;

    init_activation_buffers(nn, input_data);
    init_gradients_buffers(nn, batch_size);
    init_target_buffer(target_output);

    for(size_t epoch = 0; epoch < epoch_count; epoch++) {
        // Forward
        for (size_t l = 0; l < nn.layers_count - 1; l++) {
            unsigned in_size = nn.layers_sizes[l];
            unsigned out_size = nn.layers_sizes[l + 1];
            // Bind buffers
            rlBindShaderBuffer(activations_ssbos[l], 0);       
            rlBindShaderBuffer(activations_ssbos[l + 1], 1);  
            rlBindShaderBuffer(weights_ssbos[l], 2);         
            rlBindShaderBuffer(biases_ssbos[l], 3);         
            // Set variables
            rlSetUniform(u_is_backprop, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_increase, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_apply, &(int){0}, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_input_size, &in_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_output_size, &out_size, RL_SHADER_UNIFORM_UINT, 1);
            // Dispatch
            size_t total_threads = batch_size * nn.layers_sizes[l + 1];
            size_t groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        // COMPUTE LOSS 
        // Bind buffers
        rlBindShaderBuffer(activations_ssbos[nn.layers_count-1], 1);
        rlBindShaderBuffer(target_ssbo, 4);           
        rlBindShaderBuffer(gradients_ssbos[nn.layers_count-2], 5);     
        // Set variable
        rlSetUniform(u_calculate_loss, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            // Dispatch
            size_t total_threads = target_output.rows * target_output.cols;
            size_t groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        rlSetUniform(u_calculate_loss, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // Train
        for (ssize_t l = (ssize_t)nn.layers_count - 2; l >= 0; l--) {
            unsigned in_size = nn.layers_sizes[l];
            unsigned out_size = nn.layers_sizes[l + 1];

            // BACKPROP
            // Bind buffers
            rlBindShaderBuffer(activations_ssbos[l], 0);
            rlBindShaderBuffer(activations_ssbos[l + 1], 1);   
            rlBindShaderBuffer(weights_ssbos[l], 2);          
            rlBindShaderBuffer(biases_ssbos[l], 3);          
            rlBindShaderBuffer(gradients_ssbos[l], 5);     
            if(l > 0) 
                rlBindShaderBuffer(gradients_ssbos[l-1], 6);      
            rlBindShaderBuffer(w_gradients_ssbos[l], 7);  
            rlBindShaderBuffer(b_gradients_ssbos[l], 8); 
            // Set variables
            rlSetUniform(u_is_backprop, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_increase, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_apply, &(int){0}, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_input_size, &in_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_output_size, &out_size, RL_SHADER_UNIFORM_UINT, 1);
            // Dispatch
            size_t total_threads = (size_t)batch_size * (2 * out_size + in_size);
            size_t groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // INCREASE GRADIENTS 
            // Bind buffers
            rlBindShaderBuffer(w_gradients_total_ssbos[l], 9);  
            rlBindShaderBuffer(b_gradients_total_ssbos[l], 10); 
            // Set variables
            rlSetUniform(u_is_backprop, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_increase, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_apply, &(int){0}, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            // Dispatch
            total_threads = (unsigned)(in_size * out_size + out_size);
            groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        // Apply gradients 
        for (ssize_t l = (ssize_t)nn.layers_count - 2; l >= 0; l--) {
            size_t in_size = nn.layers_sizes[l];
            size_t out_size = nn.layers_sizes[l + 1];
            // Bind buffers
            rlBindShaderBuffer(weights_ssbos[l], 2);          
            rlBindShaderBuffer(biases_ssbos[l], 3);          
            rlBindShaderBuffer(w_gradients_total_ssbos[l], 9);  
            rlBindShaderBuffer(b_gradients_total_ssbos[l], 10); 
            // Set variables
            rlSetUniform(u_is_backprop, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_increase, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_gradients_apply, &(int){1}, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_learning_rate, &learning_rate, RL_SHADER_UNIFORM_FLOAT, 1);
            // Dispatch
            size_t total_threads = in_size * out_size + out_size;
            size_t groups = (total_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            rlReadShaderBuffer(weights_ssbos[l],
                    nn.ws[l].data,
                    nn.ws[l].cols*nn.ws[l].rows*sizeof(*nn.ws->data),
                    0);

            rlReadShaderBuffer(biases_ssbos[l],
                    nn.bs[l].data,
                    nn.bs[l].cols*sizeof(*nn.ws->data),
                    0);
        }
    }
    free_activation_buffers(nn);
    free_gradients_buffers(nn);
    free_target_buffer();
}
