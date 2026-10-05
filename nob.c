#define NOB_IMPLEMENTATION
#include "nob.h"

int main(int argc, char **argv)
{
    GO_REBUILD_URSELF(argc, argv);

    if (!nob_mkdir_if_not_exists("build")) {
        nob_log(NOB_ERROR, "Could not create build directory");
        return 1;
    }

    bool run = true;

    Cmd cmd = {0};
    cmd_append(&cmd, "clang");
    cmd_append(&cmd, "-Wall");
    cmd_append(&cmd, "-Wextra");
    cmd_append(&cmd, "-lm");
    cmd_append(&cmd, "-ggdb");
    cmd_append(&cmd, "-o", "build/main");
    cmd_append(&cmd, "main.c", "./libraylib.a");
    if (!cmd_run(&cmd)) return 1;

    if (run) {
        cmd_append(&cmd, "build/main");
        if (!cmd_run(&cmd)) return 1;
    }

    return 0;
}
