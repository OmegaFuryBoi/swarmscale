#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef N
#define N      1000      // number of agents (change with -DN=... when compiling)
#endif
#ifndef STEPS
#define STEPS  1000      // time steps to simulate
#endif
#define DENSITY 0.1f     // agents per unit area (world grows with N)
#define DT     0.1f      // time step size
#define VMAX   2.0f      // max speed
#define PI     3.14159265f
#define R      5.0f      // neighbour radius
#define ALIGN  0.1f      // how strongly agents turn toward neighbours
#define RSEP       1.0f   // separation radius
#define SEPARATION 0.5f   // strength of the push away from close neighbours
#define COHESION   0.02f  // strength of the pull toward the group's centre
#ifndef SAVE_EVERY
#define SAVE_EVERY 5      // save positions every 5 steps (0 = don't save)
#endif
#define PREY      0
#define PREDATOR  1
#define NPRED    (N / 50) // the first 2% of agents are predators
#define VPRED     2.6f    // predators are a bit faster than prey
#define RFLEE     8.0f    // prey notice predators within this distance
#define RHUNT     15.0f   // predators notice prey within this distance
#define RCATCH    1.0f    // a predator this close to its prey catches it
#define FLEE      1.0f    // how hard prey turn away from a predator
#define CHASE     0.5f    // how hard predators turn toward their prey
#define MAXCAND    128    // max prey a predator considers when planning
#define NLEAD      16     // aiming angles a predator tries for each prey
#define PLAN_STEPS 200    // how far ahead (in steps) a predator simulates a chase

float W, H;              // world size, set in main from N and DENSITY
float x[N], y[N];        // positions
float vx[N], vy[N];      // velocities
float nvx[N], nvy[N];    // new velocities (computed each step)
float avg_neighbours;    // average neighbours within R
float avg_too_close;     // average neighbours within RSEP
int   type[N];           // PREY or PREDATOR
int   target[N];         // predator: index of the prey it is chasing (-1 = none)
int   num_fleeing;       // prey that currently see a predator
int   total_catches = 0;
double thread_busy[64];  // seconds each thread spent working in flock_agents

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

float max_speed(int i)
{
    return (type[i] == PREDATOR) ? VPRED : VMAX;
}

void init_agents(void)
{
    for (int i = 0; i < N; i++) {
        type[i] = (i < NPRED) ? PREDATOR : PREY;
        x[i] = rand_range(0, W);
        y[i] = rand_range(0, H);
        float angle = rand_range(0, 2 * PI);
        vx[i] = max_speed(i) * cosf(angle);
        vy[i] = max_speed(i) * sinf(angle);
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

int plan_hunt(int ncand, const int cand[], const float cdx[], const float cdy[],
              float *hx, float *hy)
{
    int best_prey = -1, best_time = PLAN_STEPS + 1;

    for (int c = 0; c < ncand; c++) {
        int j = cand[c];
        float base = atan2f(cdy[c], cdx[c]);           
        for (int k = 0; k < NLEAD; k++) {
            float angle = base + (k - NLEAD / 2) * 0.15f; 
            float ux = cosf(angle), uy = sinf(angle);
            float px = 0, py = 0;                        
            float qx = cdx[c], qy = cdy[c];              
            for (int t = 1; t <= PLAN_STEPS; t++) {      
                px += VPRED * ux * DT;
                py += VPRED * uy * DT;
                qx += vx[j] * DT;
                qy += vy[j] * DT;
                float ex = qx - px, ey = qy - py;
                if (ex * ex + ey * ey < RCATCH * RCATCH) { 
                    if (t < best_time) {
                        best_time = t;
                        best_prey = j;
                        *hx = ux;
                        *hy = uy;
                    }
                    break;
                }
            }
        }
    }
    return best_prey;
}

void flock_agents(void)
{
    long total_neighbours = 0, total_too_close = 0;
    int fleeing = 0;

    #pragma omp parallel reduction(+:total_neighbours, total_too_close, fleeing)
    {
    int tid = omp_get_thread_num();
    double t_start = omp_get_wtime();

    #pragma omp for schedule(runtime) nowait
    for (int i = 0; i < N; i++) {
        float sum_vx = 0, sum_vy = 0;
        float sum_dx = 0, sum_dy = 0;
        float sep_x = 0, sep_y = 0;
        int count = 0;

        float best_d2 = (type[i] == PREY) ? RFLEE * RFLEE : RHUNT * RHUNT;
        float best_dx = 0, best_dy = 0;
        int   best_j  = -1;
        int   ncand = 0;                   
        int   cand[MAXCAND];
        float cdx[MAXCAND], cdy[MAXCAND];

        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            float dx = wrap_delta(x[j] - x[i], W);
            float dy = wrap_delta(y[j] - y[i], H);
            float d2 = dx * dx + dy * dy;

            if (type[i] != type[j]) {
                if (d2 < best_d2) {
                    best_d2 = d2; best_dx = dx; best_dy = dy; best_j = j;
                }
                if (type[i] == PREDATOR && d2 < RHUNT * RHUNT && ncand < MAXCAND) {
                    cand[ncand] = j; cdx[ncand] = dx; cdy[ncand] = dy;
                    ncand++;
                }
            } else if (type[i] == PREY && d2 < R * R) {  
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

        if (type[i] == PREY) {
            if (count > 0) {                  
                float avg_vx = sum_vx / count, avg_vy = sum_vy / count;
                nvx[i] += ALIGN * (avg_vx - vx[i]);
                nvy[i] += ALIGN * (avg_vy - vy[i]);
                nvx[i] += COHESION * (sum_dx / count);
                nvy[i] += COHESION * (sum_dy / count);
                nvx[i] += SEPARATION * sep_x;
                nvy[i] += SEPARATION * sep_y;
            }
            if (best_j >= 0) {                    
                float d = sqrtf(best_d2);
                nvx[i] -= FLEE * VMAX * best_dx / d;
                nvy[i] -= FLEE * VMAX * best_dy / d;
                fleeing++;
            }
            total_neighbours += count;
        } else {                               
            float hx, hy;
            int planned = plan_hunt(ncand, cand, cdx, cdy, &hx, &hy);
            if (planned >= 0) {                   
                target[i] = planned;
                nvx[i] += CHASE * VPRED * hx;
                nvy[i] += CHASE * VPRED * hy;
            } else {                              
                target[i] = best_j;
                if (best_j >= 0) {
                    float d = sqrtf(best_d2);
                    nvx[i] += CHASE * VPRED * best_dx / d;
                    nvy[i] += CHASE * VPRED * best_dy / d;
                }
            }
        }

        float speed = sqrtf(nvx[i] * nvx[i] + nvy[i] * nvy[i]);
        if (speed > 0) {                          
            nvx[i] = nvx[i] / speed * max_speed(i);
            nvy[i] = nvy[i] / speed * max_speed(i);
        }
    }

    thread_busy[tid] += omp_get_wtime() - t_start;  
    }   

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        vx[i] = nvx[i];
        vy[i] = nvy[i];
    }

    num_fleeing    = fleeing;
    avg_neighbours = (float)total_neighbours / (N - NPRED);
    avg_too_close  = (float)total_too_close / (N - NPRED);
}

int catch_prey(void)
{
    int caught = 0;
    for (int i = 0; i < NPRED; i++) {
        int j = target[i];
        if (j < 0) continue;
        float dx = wrap_delta(x[j] - x[i], W);
        float dy = wrap_delta(y[j] - y[i], H);
        if (dx * dx + dy * dy < RCATCH * RCATCH) {
            x[j] = rand_range(0, W);
            y[j] = rand_range(0, H);
            caught++;
        }
    }
    return caught;
}

void save_frame(FILE *f, int step)
{
    for (int i = 0; i < N; i++)
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%d\n", step, x[i], y[i], vx[i], vy[i], type[i]);
}

float polarization(void)
{
    float sx = 0, sy = 0;
    for (int i = NPRED; i < N; i++) {
        sx += vx[i];
        sy += vy[i];
    }
    return sqrtf(sx * sx + sy * sy) / ((N - NPRED) * VMAX);
}

int main(void)
{
    W = H = sqrtf(N / DENSITY); 
    printf("N = %d agents, world %.1f x %.1f, %d steps, %d threads\n",
           N, W, H, STEPS, omp_get_max_threads());

    srand(42);
    init_agents();

    FILE *out = NULL;
    if (SAVE_EVERY > 0) {
        out = fopen("frames.csv", "w");
        if (out == NULL) {
            printf("could not open frames.csv\n");
            return 1;
        }
        fprintf(out, "step,x,y,vx,vy,type\n");
    }

    double flock_time = 0;               
    double start = omp_get_wtime();

    for (int step = 0; step < STEPS; step++) {
        double t0 = omp_get_wtime();
        flock_agents();
        flock_time += omp_get_wtime() - t0;

        total_catches += catch_prey();
        move_agents();
        if (SAVE_EVERY > 0 && step % SAVE_EVERY == 0)
            save_frame(out, step);
        if (step % 100 == 0)
            printf("step %4d: polarization %.3f | avg neighbours %5.1f | fleeing %3d | catches %d\n",
                   step, polarization(), avg_neighbours, num_fleeing, total_catches);
    }
    double total_time = omp_get_wtime() - start;

    printf("final polarization: %.3f\n", polarization());
    printf("total time      : %.3f s (%.2f ms per step)\n", total_time, 1000 * total_time / STEPS);
    printf("flock_agents    : %.3f s (%.1f%% of total)\n", flock_time, 100 * flock_time / total_time);
    
    int nthreads = omp_get_max_threads();
    double max_busy = 0, sum_busy = 0;
    for (int t = 0; t < nthreads; t++) {
        if (thread_busy[t] > max_busy) max_busy = thread_busy[t];
        sum_busy += thread_busy[t];
    }
    printf("thread busy time:\n");
    for (int t = 0; t < nthreads; t++)
        printf("  thread %2d: %6.3f s  (%5.1f%% of the busiest)\n",
               t, thread_busy[t], 100 * thread_busy[t] / max_busy);
    printf("load imbalance  : %.2f (busiest / average, 1.00 = perfect)\n",
           max_busy / (sum_busy / nthreads));
    if (out != NULL) {
        fclose(out);
        printf("saved frames.csv\n");
    }
    return 0;
}