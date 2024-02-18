#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct ComplexNumber {
    double r;
    double i;
};

int cal_pixel(struct ComplexNumber c) {
    double z_r = 0, z_i = 0;
    double z_r2, z_i2, len;
    int j = 0;
    do {
        z_r2 = z_r * z_r;
        z_i2 = z_i * z_i;
        z_i = 2 * z_r * z_i + c.i;
        z_r = z_r2 - z_i2 + c.r;
        len = z_r2 + z_i2;
        j++;
    } while ((j < MAX_ITER) && (len < 4.0));
    return j;
}

void save_pgm(const char *file, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    pgmimg = fopen(file, "wb");
    fprintf(pgmimg, "P2\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t1 = MPI_Wtime(); 

    int rows_per_process = HEIGHT / size;
    int remainder = HEIGHT % size;
    int start = rank * rows_per_process;
    int end = start + rows_per_process - 1;
    if (rank == size - 1) {
        end += remainder; 
    }

    struct ComplexNumber c;
    int *image = malloc(WIDTH * (end - start + 1) * sizeof(int));
    int k = 0;
    for (int i = start; i <= end; i++) {
        for (int j = 0; j < WIDTH; j++) {
            c.r = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.i = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            image[k++] = cal_pixel(c);
        }
    }

    int *FullImage = NULL;
    if (rank == 0) {
        full_image = malloc(WIDTH * HEIGHT * sizeof(int));
    }

  
    MPI_Gather(image, WIDTH * (end - start + 1), MPI_INT,
               FullImage, WIDTH * rows_per_process, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (remainder > 0) {
            MPI_Recv(FullImage + WIDTH * (HEIGHT - remainder), WIDTH * remainder, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        save_pgm("mandelbrot_parallel_visualization.pgm", (int(*)[WIDTH])FullImage);
        free(FullImage);

         MPI_Barrier(MPI_COMM_WORLD); 
        double t2 = MPI_Wtime(); 

    if (rank == 0) { 
        printf("Total time taken: %f seconds\n", t2 - t1);
        }

    }

    free(image);
    MPI_Finalize();
    return 0;
}
