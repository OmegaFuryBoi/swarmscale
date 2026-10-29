#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N      1000      // number of agents
#define W      100.0f    // world width
#define H      100.0f    // world height
#define STEPS  100       // time steps to simulate
#define DT     0.1f      // time step size
#define VMAX   2.0f      // max speed
#define PI     3.14159265f

float x[N], y[N];        // positions
float vx[N], vy[N];      // velocities

// random float between lo and hi
float rand_range(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

// keep a coordinate inside [0, size) - wrap around the edges
float wrap(float v, float size)
{
    if (v >= size) v -= size;
    if (v < 0)     v += size;
    return v;
}

void init_agents(void)
{
    for (int i = 0; i < N; i++) {
        x[i] = rand_range(0, W);
        y[i] = rand_range(0, H);
        float angle = rand_range(0, 2 * PI);   // random direction
        vx[i] = VMAX * cosf(angle);
        vy[i] = VMAX * sinf(angle);
    }
}

void move_agents(void)
{
    for (int i = 0; i < N; i++) {
        x[i] = wrap(x[i] + vx[i] * DT, W);
        y[i] = wrap(y[i] + vy[i] * DT, H);
    }
}

int main(void)
{
    srand(42);           // fixed seed -> same result every run
    init_agents();

    for (int step = 0; step < STEPS; step++) {
        move_agents();
        if (step % 10 == 0)
            printf("step %3d: agent 0 at (%6.2f, %6.2f)\n", step, x[0], y[0]);
    }
    return 0;
}