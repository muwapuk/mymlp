#include <assert.h>
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

#include "darray.h"
#include "nn.h"

#define WINDOW_SIZE_W 1920
#define WINDOW_SIZE_H 1080

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
