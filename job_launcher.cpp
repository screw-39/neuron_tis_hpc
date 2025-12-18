#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <unistd.h>
using namespace std;

#include <omp.h>
#include <mpi.h>

/* Master-Worker pattern: Rank 0 acts purely as Master.
 * Other ranks use all NCPU threads as workers.
 * This eliminates thread contention on Rank 0's MPI calls.
 */
inline int uniq_tag(int nt, int rank) { return rank*1000 + nt; };
int ROOT_NODE = 0;

void launch_provider(int n_worker, const int total_work) {
    printf("Launch master for %d workers.\n", n_worker);

    int buf[1], len, ierr;
    MPI_Status status;
    char str[512];
    int next_eid = 0;
    int count = 0;

    while(1) {

        double t0 = MPI_Wtime();   // Start timing Master receive

        ierr = MPI_Recv(buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        double t1 = MPI_Wtime();   // Recv finished

        int to_source = status.MPI_SOURCE;
        int to_tag    = status.MPI_TAG;

        // Print Master communication delay
        double recv_us = (t1 - t0) * 1e6;
        printf("[MASTER] Recv from rank %d tag %d : %.2f us\n",
               to_source, to_tag, recv_us);

        if (next_eid >= total_work) {
            count++;
            buf[0] = -1;
        } else {
            buf[0] = next_eid++;
        }

        double t2 = MPI_Wtime();   // Start timing Master send

        ierr = MPI_Send(buf, 1, MPI_INT, to_source, to_tag, MPI_COMM_WORLD);

        double t3 = MPI_Wtime();   // Send finished

        double send_us = (t3 - t2) * 1e6;
        printf("[MASTER] Send to rank %d tag %d : %.2f us\n",
               to_source, to_tag, send_us);

        if (count >= n_worker) break;
    }

    printf("[Done] No more worker - Master exit.\n");
};


int launch_worker(int nt, int rank) {
    MPI_Status status;
    int bsend[1] = {0};
    int brecv[1];
    int len, ierr, tag;
    char str[1024];

    while(1) {

        tag = uniq_tag(nt, rank);

        // Only thread 0 prints timing
        double t0 = MPI_Wtime();   // Worker send → Master

        MPI_Send(bsend, 1, MPI_INT, ROOT_NODE, tag, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        ierr = MPI_Recv(brecv, 1, MPI_INT, ROOT_NODE, tag, MPI_COMM_WORLD, &status);

        double t2 = MPI_Wtime();   // Worker receive finished

        if (nt == 0) {
            printf("[WORKER %d-%d] Send: %.2f us, Recv: %.2f us, RTT: %.2f us\n",
                rank, nt,
                (t1 - t0) * 1e6,
                (t2 - t1) * 1e6,
                (t2 - t0) * 1e6);
        }

        if (ierr != MPI_SUCCESS) {
            MPI_Error_string(ierr, str, &len);
            printf("** Worker %d-%d fail : %s\n", rank, nt, str);
            continue;
        }

        int eid = brecv[0];
        if (eid < 0) break;

        snprintf(str, sizeof(str), "(uv run main.py %d %d)", tag, eid);
        system(str);
        fflush(stdout);
    }

    printf("No jobs from master - worker %d-%d exit.\n", rank, nt);
    return 0;
};


int main(int argc, char *argv[]){

    if (argc != 2) {
        printf("Usage: %s TOTAL_WORK\n", argv[0]);
        return 1;
    }

    // parse total work (must be positive integer)
    char *endptr = nullptr;
    long total_work_l = strtol(argv[1], &endptr, 10);
    if (endptr == argv[1] || total_work_l <= 0) {
        fprintf(stderr, "Invalid TOTAL_WORK: %s\n", argv[1]);
        return 1;
    }
    int total_work = (int) total_work_l;
 
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 強制檢查 MPI 執行緒安全等級
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            fprintf(stderr, "Error: MPI provided level %d, but MPI_THREAD_MULTIPLE requested.\n", provided);
        }
        MPI_Finalize();
        return 1;
    }

    // detect number of CPU cores available on this node
    int NCPU = (int) sysconf(_SC_NPROCESSORS_ONLN);
    // if (NCPU > 0) NCPU /= size; 
    if (NCPU > 0) NCPU = 30; 
    if (NCPU <= 0) NCPU = omp_get_num_procs() / size;
    if (NCPU <= 0) NCPU = 1;

    // compute how many local worker threads this rank will have
    int local_workers = 0;
    if (rank == 0) {
        // 【關鍵修改】：Rank 0 專職 Master，不產生 Worker
        local_workers = 0;
    } else {
        // 其他 Rank 的所有 NCPU 都作為 Worker
        local_workers = NCPU;
    }

    // compute total number of workers across all ranks
    int n_worker = 0;
    MPI_Allreduce(&local_workers, &n_worker, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        // 【關鍵修改】：Rank 0 直接啟動 Master，不再使用 OpenMP parallel
        // 確保 Master 在主執行緒中運行，消除競爭
        launch_provider(n_worker, total_work);
    } else {
        // 其他 ranks: 啟動 NCPU 執行緒，全部都是 Workers
        #pragma omp parallel num_threads(NCPU)
        {
            int nt = omp_get_thread_num();
            launch_worker(nt, rank);
        }
    }

    MPI_Finalize();
    return 0;
}