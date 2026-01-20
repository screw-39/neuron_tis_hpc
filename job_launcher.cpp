#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <set>
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
    set<int> stopped_workers;

    while(1) {
        ierr = MPI_Recv(buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int to_source = status.MPI_SOURCE;
        int to_tag    = status.MPI_TAG;

        if (next_eid >= total_work) {
            if (!stopped_workers.count(to_tag)) {
                stopped_workers.insert(to_tag);
                count++;
            }
            buf[0] = -1;
        } else {
            buf[0] = next_eid++;
        }

        ierr = MPI_Send(buf, 1, MPI_INT, to_source, to_tag, MPI_COMM_WORLD);

        if (count >= n_worker) break;
    }

    printf("[Done] No more worker - Master exit.\n");
};


int launch_worker(int nt, int rank) {
    MPI_Status status;
    int bsend[1] = {0}; // dummy request; payload unused
    int brecv[1];
    int len, ierr;
    char str[1024];
    int worker_id = uniq_tag(nt, rank);
    int tag = worker_id;

    while(1) {
        MPI_Send(bsend, 1, MPI_INT, ROOT_NODE, tag, MPI_COMM_WORLD);

        ierr = MPI_Recv(brecv, 1, MPI_INT, ROOT_NODE, tag, MPI_COMM_WORLD, &status);

        if (ierr != MPI_SUCCESS) {
            MPI_Error_string(ierr, str, &len);
            printf("** Worker %d-%d fail : %s\n", rank, nt, str);
            continue;
        }

        int eid = brecv[0];
        if (eid < 0) break;

        snprintf(
            str, sizeof(str),
            "(uv run main.py %d > worker_%d.log 2>&1)",
            worker_id, worker_id
        );
        system(str);
    }

    printf("No jobs from master - worker %d-%d exit.\n", rank, nt);
    return 0;
};


int main(int argc, char *argv[]){

    if (argc != 3) {
        printf("Usage: %s TOTAL_WORK WORKERS_PER_RANK\n", argv[0]);
        return 1;
    }

    // parse total work (must be positive integer)
    int total_work = atoi(argv[1]);
    int workers_per_rank = atoi(argv[2]);

    if (total_work <= 0 || workers_per_rank <= 0) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }
    
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (workers_per_rank >= 1000 && rank == 0) {
        fprintf(stderr, "workers_per_rank must be < 1000\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 強制檢查 MPI 執行緒安全等級
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            fprintf(stderr, "Error: MPI provided level %d, but MPI_THREAD_MULTIPLE requested.\n", provided);
        }
        MPI_Finalize();
        return 1;
    }

    // detect number of CPU cores available on this node
    int max_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (max_cpu <= 0) max_cpu = omp_get_num_procs();
    if (max_cpu <= 0) max_cpu = 1;

    // 保護：不要超過節點實際 CPU
    int NCPU = min(workers_per_rank, max_cpu);

    // compute how many local worker threads this rank will have
    int local_workers = (rank == 0) ? 0 : NCPU;
    int n_worker = 0;

    // compute total number of workers across all ranks
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