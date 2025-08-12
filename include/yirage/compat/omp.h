#ifndef OMP_COMPAT_H
#define OMP_COMPAT_H

/*
 * OpenMP compatibility header for non-OpenMP builds
 * Provides stub implementations when OpenMP is not available
 */

#ifdef _OPENMP
// If OpenMP is available, include the real header
#include <omp.h>
#else
// OpenMP compatibility stubs for non-OpenMP builds

#ifdef __cplusplus
extern "C" {
#endif

// OpenMP API function stubs
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_procs(void) { return 1; }
static inline void omp_set_num_threads(int num_threads) { (void)num_threads; }
static inline int omp_in_parallel(void) { return 0; }
static inline void omp_set_dynamic(int dynamic_threads) { (void)dynamic_threads; }
static inline int omp_get_dynamic(void) { return 0; }
static inline void omp_set_nested(int nested) { (void)nested; }
static inline int omp_get_nested(void) { return 0; }

// Lock functions (no-op implementations)
typedef struct { int dummy; } omp_lock_t;
typedef struct { int dummy; } omp_nest_lock_t;

static inline void omp_init_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_destroy_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_set_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_unset_lock(omp_lock_t *lock) { (void)lock; }
static inline int omp_test_lock(omp_lock_t *lock) { (void)lock; return 1; }

static inline void omp_init_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_destroy_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_set_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_unset_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline int omp_test_nest_lock(omp_nest_lock_t *lock) { (void)lock; return 1; }

// Timing functions
static inline double omp_get_wtime(void) {
    // Simple fallback using standard C library
    #include <time.h>
    return (double)clock() / CLOCKS_PER_SEC;
}

static inline double omp_get_wtick(void) {
    return 1.0 / CLOCKS_PER_SEC;
}

#ifdef __cplusplus
}
#endif

// OpenMP pragma compatibility macros
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_CRITICAL
#define OMP_SINGLE
#define OMP_MASTER
#define OMP_BARRIER
#define OMP_ATOMIC

// Pragma replacement macros (these become no-ops)
#ifndef _OPENMP
// Note: _Pragma is a builtin macro, so we don't redefine it
// Instead we handle pragma directives case by case if needed
#endif

#endif /* _OPENMP */

#endif /* OMP_COMPAT_H */
