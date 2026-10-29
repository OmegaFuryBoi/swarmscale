#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N      1000      // number of agents
#define W      100.0f    // world width
#define H      100.0f    // world height
#define STEPS  1000      // time steps to simulate
#define DT     0.1f      // time step size
#define VMAX   2.0f      // max speed
#define PI     3.14159265f
#define R      5.0f      // neighbour radius
#define ALIGN  0.1f      // how strongly agents turn toward neighbours

float x[N], y[N];        // positions
float vx[N], vy[N];      // velocities
float nvx[N], nvy[N];    // new velocities (computed each step)

float rand_range(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

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
        float angle = rand_range(0, 2 * PI);  
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

float wrap_delta(float d, float size)
{
    if (d >  size / 2) d -= size;
    if (d < -size / 2) d += size;
    return d;
}

void align_agents(void)
{
    for (int i = 0; i < N; i++) {
        float sum_vx = 0, sum_vy = 0;
        int count = 0;

        for (int j = 0; j < N; j++) {            
            if (j == i) continue;
            float dx = wrap_delta(x[j] - x[i], W);
            float dy = wrap_delta(y[j] - y[i], H);
            if (dx * dx + dy * dy < R * R) {      
                sum_vx += vx[j];
                sum_vy += vy[j];
                count++;
            }
        }

        nvx[i] = vx[i];
        nvy[i] = vy[i];
        if (count > 0) {
            float avg_vx = sum_vx / count, avg_vy = sum_vy / count;
            nvx[i] += ALIGN * (avg_vx - vx[i]);   
            nvy[i] += ALIGN * (avg_vy - vy[i]);
        }

        float speed = sqrtf(nvx[i] * nvx[i] + nvy[i] * nvy[i]);
        if (speed > 0) {                          
            nvx[i] = nvx[i] / speed * VMAX;
            nvy[i] = nvy[i] / speed * VMAX;
        }
    }

    for (int i = 0; i < N; i++) {                
        vx[i] = nvx[i];
        vy[i] = nvy[i];
    }
}

float polarization(void)
{
    float sx = 0, sy = 0;
    for (int i = 0; i < N; i++) {
        sx += vx[i];
        sy += vy[i];
    }
    return sqrtf(sx * sx + sy * sy) / (N * VMAX);
}

int main(void)
{
    srand(42);           
    init_agents();

    for (int step = 0; step < STEPS; step++) {
        align_agents();
        move_agents();
        if (step % 100 == 0)
            printf("step %3d: polarization %.3f\n", step, polarization());
    }
    printf("final polarization: %.3f\n", polarization());

    return 0;
}