#include "test.h"

void print_test(double* array1, int array1m, int array1n,
                double* array2, int array2m, int array2n)
{

  std::cout << " array1 " << array1m  << " " << array1n << std::endl;
  for(int i = 0; i < array1n * array1m; ++i)
    std::cout << array1[i] << std::endl;

  std::cout << " array2 " << array2m  << " " << array2n << std::endl;
  for(int i = 0; i < array2n * array2m; ++i)
    std::cout << array2[i] << std::endl;


}
