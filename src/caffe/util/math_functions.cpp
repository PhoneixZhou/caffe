#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}
template <>
int caffe_cpu_asum<int>(const int n,const int* x){
  int sum = 0;
  for(int i = 0;i<n;i++){
    sum+= abs(x[i]);
  }
  return sum;
}


template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

/*************************start of ssl methods************************/
template <>
void caffe_cpu_sparse_dense2csr<float>(const int M, const int N,
     float* A,
     float* A_nonzero_buf, int* A_nonzero_idx_buf,int * A_idx_pointer_buf){
      #ifdef USE_MKL 
      MKL_INT info;
      const MKL_INT job[] = {0,0,0,2,M*N,1};
      mkl_sdnscsr(job,&M,&N,A,
        &N,A_nonzero_buf,A_nonzero_idx_buf,A_idx_pointer_buf, &info);
      if(info){
        LOG(FATAL)<<"The routine is interrupted processing the "<<info<<"-th row "
        <<"because there is no space in the arrays acsr and ja according to the value nzmax.";
      }
      #else 
      NOT_IMPLEMENTED;
      #endif 
     }
template <>
void caffe_cpu_sparse_dense2csr<double>(const int M, const int N,
    double* A,
    double* A_nonzero_buf,int* A_nonzero_idx_buf,int* A_idx_pointer_buf){
      #ifdef USE_MKL
      MKL_INT info;
      const MKL_INT job[] = {0,0,0,2,M*N,1};
      mkl_ddnscsr(job,&M,&N,A, 
        &N,A_nonzero_buf,A_nonzero_idx_buf,A_idx_pointer_buf,&info);
      if(info){
        LOG(FATAL)<<"The routine is interrupted prcessing the "<<
            info <<"-th row "
            <<"because there is non space in the arrays acsr and ja according to the value nzmax.";
      }
      #else
      NOT_IMPLEMENTED; 
      #endif 
    }

template <typename Dtype>
void caffe_cpu_if_all_zero(const int M, const int N, const Dtype* x,int *y,bool dimen){
  if(dimen){//along columns
    for(int col=0;col<N;++col){
      y[col] = true;
      for(int row = 0;row<M;row++){
        if(x[col+row * N] !=0){
          y[col] = false;
          break;
        }
      }
    }
  }else{//along rows
    for(int row = 0;row<M;++row){
      y[row] = true;
      for(int col = 0;col<N;col++){
        if(x[col + row * N] !=0){
          y[row] = false;
          break;
        }
      }
    }   
  }
}

template 
void caffe_cpu_if_all_zero(const int M,const int N, const float* x,int* y,bool dimen);
template 
void caffe_cpu_if_all_zero(const int M,const int N, const double* x,int* y,bool dimen);

template <typename Dtype>
void caffe_cpu_all_zero_mask(const int M,const int N,const Dtype* X,Dtype * Y){
  //along rows
  Dtype val = (Dtype)1;
  for(int row = 0;row<M;++row){
    val = (Dtype)0;
    for(int col = 0;col<N;col++){
      if(X[col + row * N] !=0){
        val = (Dtype)1;
        break;
      }
    }
    caffe_set(N,val,Y+row*N);
  }

  //along columns
  for(int col = 0;col < N;++col){
    val = (Dtype)0;
    for(int row = 0;row<M;row++){
      if(X[col + row *N] !=0){
        val = (Dtype)1;
        break;
      }
    }
    if(!val){
      for(int row = 0;row<M;row++){
        Y[col + row *N] = val;
      }
    }
  }
}
template 
void caffe_cpu_all_zero_mask(const int M, const int N,const float* X,float*y);
template 
void caffe_cpu_all_zero_mask(const int M, const int N,const double *X,double *y);
template 
void caffe_cpu_all_zero_mask(const int M, const int N, const int * X,int *y);
template 
void caffe_cpu_all_zero_mask(const int M, const int N,const unsigned int *X, unsigned int * y);

template <typename Dtype>
Dtype caffe_cpu_group_sparsity(const int M, const int N, const Dtype* x, bool dimen){
  Dtype sparsity = (Dtype)0;
  int counter = 0;
  if(dimen){//along columns
    for(int col = 0;col < N;++col){
      counter++;
      for(int row = 0;row<M;row++){
        if(x[col + row * N]!=0){
          counter--;
          break;
        }
      }
    }
    sparsity = (Dtype)counter / (Dtype)N;
  }else{//along rows 
    for(int row = 0;row<M;++row){
      counter++;
      for(int col = 0;col<N;col++){
        if(x[col+row * N]!=0){
          counter--;
          break;
        }
      }
    }
    sparsity = (Dtype)counter/(Dtype)M;
  }
  return sparsity;
}
template float caffe_cpu_group_sparsity(const int M, const int N, const float* x,bool dimen);
template double caffe_cpu_group_sparsity(const int M, const int N, const double* x, bool dimen);

template <typename Dtype>
void caffe_cpu_concatenate_rows_cols(const int M, const int N, const Dtype * x, Dtype* y,const int* col_mask, const int* row_mask){
  int left_cols = 0;
  for(int i = 0;i<N;i++){
    left_cols += !col_mask[i];
  }

  int cur_row = 0;
  for(int row = 0;row<M;row++){
    if(!row_mask[row]){
      int cur_col = 0;
      for(int col = 0;col<N;col++){
        if(!col_mask[col]){
          y[cur_row * left_cols + cur_col] = x[row * N + col];
          cur_col++;
        }
      }
      CHECK_EQ(cur_col,left_cols);
      cur_row++;
    }
  }
}

template void caffe_cpu_concatenate_rows_cols<float>(const int M,const int N,const float* x,float* y,const int*col_mask,const int* row_mask);
template void caffe_cpu_concatenate_rows_cols<double>(const int M,const int N,const double * x,double * y, const int* col_mask,const int* row_mask);
/************************end of ssl methods***************************/

}  // namespace caffe
