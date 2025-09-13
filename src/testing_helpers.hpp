#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>

template<typename T>
int cmpArrays(int n, T* a, T* b)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != b[i])
        {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

inline void printDesc(const char* desc)
{
    printf("==== %s ====\n", desc);
}

template<typename T>
inline void printCmpResult(int n, T* a, T* b)
{
    printf("    %s \n", cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
}

template<typename T>
inline void printCmpLenResult(int n, int expN, T* a, T* b)
{
    if (n != expN)
    {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    printf("    %s \n",
           (n == -1 || n != expN) ? "FAIL COUNT"
           : cmpArrays(n, a, b)   ? "FAIL VALUE"
                                  : "passed");
}

inline void zeroArray(int n, int* a)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = 0;
    }
}

inline void onesArray(int n, int* a)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = 1;
    }
}

inline void genArray(int n, int* a, int maxval)
{
    srand(time(nullptr));

    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % maxval;
    }
}

inline void genConsecutiveArray(int n, int* a)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
}

inline void printArray(int n, int* a, bool abridged = false)
{
    printf("    [ ");
    for (int i = 0; i < n; i++)
    {
        if (abridged && i + 2 == 15 && n > 16)
        {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("] - count: ");
    printf("%d\n", n);
}

template<typename T>
inline void printElapsedTime(T time, const char* note = "")
{
    std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}
