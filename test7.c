#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N 700
#define MAX_ITER 1000

int main()
{
    int A[N][N];
    double x[N];
    double y[N];
    double tol = 0.000001;
    double lam_0 = 0.0;
    double lam_1 = 0.0;
    int i, j, k, iter;
    // define jumlah thread yang digunakan
    int num_threads = 12;
    clock_t start_time, end_time;
    double cpu_time_used;
    omp_set_num_threads(num_threads);
    // mencatat waktu awal sebelum algoritma dimulai
    start_time = clock();

// membentuk matrix ukuran nxn dengan nilai random
#pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = rand();
        }
        x[i] = 1.0;
    }

    // mulai algoritma
    for (iter = 0; iter < MAX_ITER; iter++)
    {
// y=A*x
#pragma omp parallel for private(i, j)
        for (i = 0; i < N; i++)
        {
            y[i] = 0.0;
            for (j = 0; j < N; j++)
            {
#pragma omp atomic
                y[i] += A[i][j] * x[j];
            }
        }
        // yp = max(|y|)
        int n = sizeof(y) / sizeof(y[0]);
        double yp = fabs(y[0]);
#pragma omp parallel for private(i) shared(yp)
        for (i = 1; i < n; i++)
        {
            double abs_val = fabs(y[i]);
#pragma omp critical
            {
                if (abs_val > yp)
                {
                    yp = abs_val;
                }
            }
        }
        // lam=yp
        double lam = yp;
        // mempercepat konvergensi menggunakan aitken method
        double lam_aitken = lam_0 - ((lam_1 - lam_0) * (lam_1 - lam_0) / (lam - 2 * lam_1 + lam_0));
        printf("Approx Dominant Eigenvalue: %lf\n", lam_aitken);
        // jika yp=0 stop
        if (yp == 0)
        {
            break;
        }
        // menghitung error
        double arr[N];
        int f = sizeof(arr) / sizeof(arr[0]);
#pragma omp parallel for private(i) shared(yp)
        for (i = 0; i < f; i++)
        {
            arr[i] = fabs(x[i] - (y[i] / yp));
        }

        double err = arr[0];

#pragma omp parallel for private(i) shared(err)
        for (i = 1; i < f; i++)
        {
            double abs_val_1 = arr[i];
#pragma omp critical
            {
                if (abs_val_1 > err)
                {
                    err = abs_val_1;
                }
            }
        }
// mengupdate vektor x
#pragma omp parallel for private(i) shared(yp)
        for (i = 0; i < N; i++)
        {
            x[i] = y[i] / yp;
            printf("%lf\n", x[i]);
        }
        printf("\n");
        // print error
        printf("Error: %lf\n", err);
        // jika error < tol, stop algoritma
        if (err < tol && iter >= 4)
        {
            break;
        }
        lam_0 = lam_1;
        lam_1 = lam;
    }
    printf("Iteration Count: %d\n", iter + 1);
    // memcatat waktu akhir
    end_time = clock();
    // menghitung running time berdasarkan waktu akhir dan waktu awal
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("-----------------------------------\n");
    printf("CPU time used: %lf seconds", cpu_time_used);
    return 0;
}