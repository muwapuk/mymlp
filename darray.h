#ifndef DARRAY_H
#define DARRAY_H

#define ARRAY_LEN(xs) (sizeof((xs))/sizeof((xs)[0]))
#define da_append(da, item)\
    do {\
        if((da).count >= (da).capacity) {\
            if((da).capacity == 0) (da).capacity = 8;\
            else (da).capacity*=2;\
            void *new_items = realloc((da).items, (da).capacity*sizeof(*(da).items));\
            if(!new_items) {\
                fprintf(stderr, "Error: realloc failed in da_append\n");\
                exit(1);\
            }\
            (da).items = new_items;\
        }\
        (da).items[(da).count++] = item;\
    } while(0)
#define da_at(da, i)(da).items[i]

#endif
