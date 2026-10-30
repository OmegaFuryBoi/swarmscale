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
#define RSEP       1.0f   // separation radius
#define SEPARATION 0.5f   // strength of the push away from close neighbours
#define COHESION   0.02f  // strength of the pull toward the group's centre
#define SAVE_EVERY 5      // save positions every 5 steps for the animation

float x[N], y[N];        // positions
float vx[N], vy[N];      // velocities
float nvx[N], nvy[N];    // new velocities (computed each step)
float avg_neighbours;    // average neighbours within R
float avg_too_close;     // average neighbours within RSEP

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

void flock_agents(void)
{
    long total_neighbours = 0, total_too_close = 0;         

    for (int i = 0; i < N; i++) {
        float sum_vx = 0, sum_vy = 0;     
        float sum_dx = 0, sum_dy = 0;     
        float sep_x = 0, sep_y = 0;       
        int count = 0;

        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            float dx = wrap_delta(x[j] - x[i], W);
            float dy = wrap_delta(y[j] - y[i], H);
            float d2 = dx * dx + dy * dy;
            if (d2 < R * R) {
                sum_vx += vx[j];
                sum_vy += vy[j];
                sum_dx += dx;                                  
                sum_dy += dy;                                  
                count++;
                if (d2 < RSEP * RSEP && d2 > 0) {              
                    sep_x -= dx / d2;     
                    sep_y -= dy / d2;
                    total_too_close++;
                }
            }
        }

        nvx[i] = vx[i];
        nvy[i] = vy[i];
        if (count > 0) {
            float avg_vx = sum_vx / count, avg_vy = sum_vy / count;
            nvx[i] += ALIGN * (avg_vx - vx[i]);                
            nvy[i] += ALIGN * (avg_vy - vy[i]);
            nvx[i] += COHESION * (sum_dx / count);             
            nvy[i] += COHESION * (sum_dy / count);
            nvx[i] += SEPARATION * sep_x;                     
            nvy[i] += SEPARATION * sep_y;
        }
        total_neighbours += count;                             

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

    avg_neighbours = (float)total_neighbours / N;             
    avg_too_close  = (float)total_too_close / N;               
}

void save_frame(FILE *f, int step)
{
    for (int i = 0; i < N; i++)
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f\n", step, x[i], y[i], vx[i], vy[i]);
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
    FILE *out = fopen("frames.csv", "w");
    if (out == NULL) {
        printf("could not open frames.csv\n");
        return 1;
    }
    fprintf(out, "step,x,y,vx,vy\n");      

    for (int step = 0; step < STEPS; step++) {
        flock_agents();
        move_agents();
        if (step % SAVE_EVERY == 0)
            save_frame(out, step);
        if (step % 100 == 0)
            printf("step %4d: polarization %.3f | avg neighbours %5.1f | too close %.2f\n",
                   step, polarization(), avg_neighbours, avg_too_close);
    }
    printf("final polarization: %.3f\n", polarization());
    fclose(out);
    printf("saved frames.csv\n");

    return 0;
}