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
    printf("\033[1;35m==== %s ====\033[0m\n", desc);  // make pink
}

template<typename T>
inline void printCmpResult(int n, T* a, T* b)
{
    printf("    %s \033[0m\n", cmpArrays(n, a, b) ? "\033[1;31mFAIL VALUE" : "\033[1;32mpassed");
}

template<typename T>
inline void printCmpLenResult(int n, int expN, T* a, T* b)
{
    if (n != expN)
    {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    printf("    %s \033[0m\n",
           (n == -1 || n != expN) ? "\033[1;31mFAIL COUNT"
           : cmpArrays(n, a, b)   ? "\033[1;31mFAIL VALUE"
                                  : "\033[1;32mpassed");
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

inline void copyArray(int n, int* copy, const int* a)
{
    for (int i = 0; i < n; i++)
    {
        copy[i] = a[i];
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
