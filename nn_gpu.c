#include "nn_gpu.h"
#include "raylib.h"
#include "rlgl.h"
#include "glad.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

bool is_weighs_biases_init = false;

unsigned int *activations_ssbos;
unsigned int *weights_ssbos;
unsigned int *biases_ssbos;
unsigned int target_buffer;
unsigned int *gradients;
unsigned int *w_gradients;
unsigned int *b_gradients;
unsigned int *w_gradients_total;
unsigned int *b_gradients_total;

int u_batch_size;
int u_input_size;
int u_output_size;

int u_is_backprop;
int u_is_last_layer;

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
    unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    UnloadFileText(nn_train_shader_src);
    nn_shader_program = rlLoadShaderProgramCompute(nn_train_shader);
    rlEnableShader(nn_shader_program);

    free(activations_ssbos);
    free(weights_ssbos);
    free(biases_ssbos);

    init_weights_and_biases_buffers(nn);
    is_weighs_biases_init  = true;

    u_batch_size = rlGetLocationUniform(nn_shader_program, "u_batch_size");
    u_input_size = rlGetLocationUniform(nn_shader_program, "u_input_size");
    u_output_size = rlGetLocationUniform(nn_shader_program, "u_output_size");
    u_is_backprop = rlGetLocationUniform(nn_shader_program, "u_is_backprop");
    u_is_last_layer = rlGetLocationUniform(nn_shader_program, "u_is_last_layer");
}
void clear_nn_gpu(Network nn)
{
    //TODO: Unload buffers

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
void nn_forward_gpu(Network nn, Mat input_data, Mat output_data, bool *should_stop)
{
    if(!is_weighs_biases_init) return;
    init_activation_buffers(nn, input_data);
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlBindShaderBuffer(activations_ssbos[l], 0);
        rlBindShaderBuffer(activations_ssbos[l+1], 1);
        rlBindShaderBuffer(weights_ssbos[l], 2);
        rlBindShaderBuffer(biases_ssbos[l], 3);

        size_t workgroups = input_data.rows*nn.layers_sizes[l+1]/WORKGROUP_SIZE;

        rlComputeShaderDispatch(workgroups, 1, 1);
    }
    rlReadShaderBuffer(
        activations_ssbos[nn.layers_count-1],
        output_data.data,
        output_data.rows*output_data.cols*sizeof(output_data.data),
        0);
} 
static void init_gradients_buffers(Network nn, size_t batch_size)
{
    unsigned int *gradients;
    unsigned int *w_gradients;
    unsigned int *b_gradients;

    gradients = malloc((nn.layers_count-1)*sizeof(*gradients));
    w_gradients = malloc((nn.layers_count-1)*sizeof(*w_gradients));
    b_gradients = malloc((nn.layers_count-1)*sizeof(*b_gradients));
    w_gradients_total = malloc((nn.layers_count-1)*sizeof(*w_gradients));
    b_gradients_total = malloc((nn.layers_count-1)*sizeof(*b_gradients));
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        gradients[l] = rlLoadShaderBuffer(
            nn.ns[l+1].cols*batch_size*sizeof(*nn.ns->data),
            NULL,
            RL_DYNAMIC_COPY);
        w_gradients[l] = rlLoadShaderBuffer(
            nn.ws[l].rows*nn.ws[l].cols*batch_size*sizeof(*nn.ws->data),
            NULL,
            RL_DYNAMIC_COPY);
        b_gradients[l] = rlLoadShaderBuffer(
            nn.bs[l].cols*batch_size*sizeof(*nn.bs->data),
            NULL,
            RL_DYNAMIC_COPY);
        w_gradients_total[l] = rlLoadShaderBuffer(
            nn.ws[l].rows*nn.ws[l].cols*sizeof(*nn.ws->data),
            NULL,
            RL_DYNAMIC_COPY);
        b_gradients_total[l] = rlLoadShaderBuffer(
            nn.bs[l].cols*sizeof(*nn.bs->data),
            NULL,
            RL_DYNAMIC_COPY);
    }

}
static void init_target_buffer(Mat target_output)
{
    target_buffer = rlLoadShaderBuffer(
        target_output.rows*target_output.cols*sizeof(*target_output.data),
        target_output.data,
        RL_DYNAMIC_COPY);
}
void nn_train_gpu(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop)
{
    if(!is_weighs_biases_init) return;

    size_t batch_size = input_data.rows;

    init_activation_buffers(nn, input_data);
    init_gradients_buffers(nn, batch_size);

    for(size_t epoch = 0; epoch < epoch_count; epoch++) {
        for (size_t l = 0; l < nn.layers_count - 1; l++) {
            rlBindShaderBuffer(activations_ssbos[l], 0);       
            rlBindShaderBuffer(activations_ssbos[l + 1], 1);  
            rlBindShaderBuffer(weights_ssbos[l], 2);         
            rlBindShaderBuffer(biases_ssbos[l], 3);         
            
            rlSetUniform(u_is_backprop, &(int){0}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_input_size, &nn.layers_sizes[l], RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_output_size, &nn.layers_sizes[l+1], RL_SHADER_UNIFORM_UINT, 1);

            size_t total_threads = batch_size * nn.layers_sizes[l + 1];
            size_t groups = (total_threads + 128 - 1) / 128;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        // train
        for (ssize_t l = (ssize_t)nn.layers_count - 2; l >= 0; l--) {
            size_t in_size = nn.layers_sizes[l];
            size_t out_size = nn.layers_sizes[l + 1];

            // Backprop
            rlBindShaderBuffer(activations_ssbos[l], 0);
            rlBindShaderBuffer(activations_ssbos[l + 1], 1);   
            rlBindShaderBuffer(weights_ssbos[l], 2);          
            rlBindShaderBuffer(biases_ssbos[l], 3);          
            rlBindShaderBuffer(target_buffer, 4);           
            rlBindShaderBuffer(gradients[l], 5);     
            if(l > 0) {
                rlBindShaderBuffer(gradients[l-1], 6);      
            } else {
                rlBindShaderBuffer(0, 6); // Layer 0 input gradients aren't needed for input data
            }
            rlBindShaderBuffer(w_gradients[l], 7);  
            rlBindShaderBuffer(b_gradients[l], 8); 

            bool is_lust_layer = (l == (ssize_t)nn.layers_count - 2) ? 1 : 0;
            rlSetUniform(u_is_backprop, &(int){1}, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_is_last_layer, &is_lust_layer, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(u_batch_size, &batch_size, RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_input_size, &nn.layers_sizes[l], RL_SHADER_UNIFORM_UINT, 1);
            rlSetUniform(u_output_size, &nn.layers_sizes[l+1], RL_SHADER_UNIFORM_UINT, 1);

            size_t max_threads = batch_size * (in_size > out_size ? in_size : out_size);
            size_t groups = (max_threads + 128 - 1) / 128;
            rlComputeShaderDispatch(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Reduction 
            // TODO: Reduction shader
        }
    }
}
