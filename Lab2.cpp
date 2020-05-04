#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"

using namespace std;

void generate(const int size)
{
    int** mat = new int* [size];
    for (auto i = 0; i < size; i++)
    {
        mat[i] = new int[size + 1];
        mat[i][i] = 1;

        for (auto j = 0; j < size + 1; j++)
        {
            if (i != j)
            {
                mat[i][j] = rand() % 100;
                mat[i][i] += mat[i][j];
            }
        }
        if (mat[i][i] == 0)
            mat[i][i]++;
    }

    ofstream out1, out2;

    out1.open("matrix" + to_string(size) + ".txt");
    out2.open("approx" + to_string(size) + ".txt");

    out1 << size << " " << size + 1 << endl;
    out2 << size << endl;

    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size + 1; j++)
            out1 << mat[i][j] << " ";
        out1 << endl;
        out2 << rand() % 10 << endl;
    }

    for (auto i = 0; i < size; i++)
        delete[] mat[i];
    delete[] mat;

    out1.close();
    out2.close();
    return;
}

void readFiles(const int size, long double* a, long double* b, long double *x)
{
    string name = "matrix" + to_string(size) + ".txt";
    cout << "Reading file " << name << endl;
    ifstream in;

    in.open(name);
    if (!in.is_open())
        throw "Can't open file " + name;

    int m, n;
    in >> m >> n;
    if (n != m + 1 || size != m)
    {
        in.close();
        throw "Incorrect matrix parameters! n = " + to_string(n) + " m = " + to_string(m);
    }

    for (auto i = 0; i < m; i++)
    {
        for (auto j = 0; j < m; j++)
            in >> a[i * m + j];
        in >> b[i];
    }
    in.close();

    name = "approx" + to_string(m) + ".txt";
    cout << "Reading file " << name << endl;
    int m1;

    in.open(name);
    if (!in.is_open())
        throw "Can't open file " + name;

    in >> m1;
    if (m1 != m)
    {
        in.close();
        throw "Incorrect length of X vector! Global m = " + to_string(m) + " Local m = " + to_string(m1);
    }
    for (auto i = 0; i < m; i++)
        in >> x[i];

    in.close();
}

bool isSolvable(const int m, const long double* a)
{
    cout << "Checking matrix " << endl;
    for (auto i = 0; i < m; i++)
    {
        double sum = 0;
        for (auto j = 0; j < m; j++)
            if (i != j)
                sum += a[i * m + j];
        if (a[i * m + i] <= sum || a[i * m + i] == 0)
        {
            cout << "Can't work with this matrix! Row " + to_string(i + 1) + " is bad";
            return false;
        }
    }
    cout << "Matrix is ok" << endl;
    return true;
}

void printResult(const int m, const long double* x)
{
    ofstream out;
    out.open("result.txt");

    out << m << endl;
    for (auto i = 0; i < m; i++)
        out << x[i] << endl;
    out.close();
}

void calculateLengthsAndOffsets(const int size, const int m, int * lengths, int* lengths_a, int *offsets, int* offsets_a)
{
    for (auto i = 0; i < size; i++)
        lengths[i] = m / size;
    for (auto i = 0; i < m - (m / size) * size; i++)
        lengths[i]++;
    if (size > m)
    {
        for (int i = m; i < size; i++)
            lengths[i] = 0;
    }
    for (auto i = 0; i < size; i++)
        lengths_a[i] = lengths[i] * m;

    offsets[0] = 0;
    offsets_a[0] = 0;
    for (auto i = 1; i < size; i++)
    {
        offsets[i] = offsets[i - 1] + lengths[i - 1];
        offsets_a[i] = offsets_a[i - 1] + lengths_a[i - 1];
    }
}

double findLocalMaxNorm(const int length, const int offset, const long double* x, const long double* temp_x)
{
    double norm = 0;
    for (auto i = 0; i < length; i++)
    {
        if (abs(x[offset + i] - temp_x[i]) > norm)
            norm = abs(x[offset + i] - temp_x[i]);
    }
    return norm;
}

double findGlobalMaxNorm(const int size, const long double* norms)
{
    double norm = norms[0];
    for (int i = 0; i < size; i++)
    {
        if (norms[i] > norm)
            norm = norms[i];
    }
    return norm;
}

void Jacobi(const int m, const int size, const int rank, const double eps,
    const int* offsets, const int* lengths,
    long double * temp_x, long double* temp_a, long double* temp_b, 
    long double* norms, long double* x)
{
    long double norm = 1;
    auto offset = offsets[rank];
    auto length = lengths[rank];

    while (norm > eps)
    {
        for (auto i = 0; i < length; i++)
        {
            temp_x[i] = temp_b[i];

            for (auto j = 0; j < m; j++)
            {
                if (i + offset != j)
                    temp_x[i] -= temp_a[i * m + j] * x[j];
            }
            temp_x[i] /= temp_a[i * m + i + offset];
        }

        norm = findLocalMaxNorm(length, offset, x, temp_x);
        
        MPI_Allgather(&norm, 1, MPI_LONG_DOUBLE, norms, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(temp_x, length, MPI_LONG_DOUBLE, x, lengths, offsets, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
        
        norm = findGlobalMaxNorm(size, norms);
    }
}

int main(int argc, char* argv[])
{
    long double* a = {}, * b = {}, * x = {};
    try
    {
        MPI_Init(&argc, &argv);
        int size, rank;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int m = 300;            // <---- put here matrix size 
        const double eps = 0.001;

        a = new long double[m * m]; 
        b = new long double[m]; 
        x = new long double[m];

        if (rank == 0)
        {
            //generate(m);
            readFiles(m, a, b, x);  
            if (!isSolvable(m, a))
                throw "Can not solve";
        }

        // prepare things for Scatterv
        int* lengths = new int[size];
        int* lengths_a = new int[size];
        int* offsets = new int[size];
        int* offsets_a = new int[size];
        calculateLengthsAndOffsets(size, m, lengths, lengths_a, offsets, offsets_a);
        auto length_a = lengths_a[rank];
        auto length = lengths[rank];
        long double* temp_a = new long double[length_a];
        long double* temp_b = new long double[length];
        long double* temp_x = new long double[length];
        long double* norms = new long double[size];

        // do Scatterv and Bcast
        MPI_Scatterv(a, lengths_a, offsets_a, MPI_LONG_DOUBLE, temp_a, length_a, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(b, lengths, offsets, MPI_LONG_DOUBLE, temp_b, length, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(x, m, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

        // Jakobi
        const auto start = MPI_Wtime();
        Jacobi(m, size, rank, eps, offsets, lengths, temp_x, temp_a, temp_b, norms, x);
        const auto finish = MPI_Wtime();

        if (rank == 0)
        {
            cout << "Number of processes: " << size << " Size: " << m << " Time: " << finish - start << endl;
            printResult(m, x);
        }

        delete[] a;
        delete[] temp_a;
        delete[] b;
        delete[] temp_b;
        delete[] x;
        delete[] temp_x;
        delete[] norms;
        delete[] lengths;
        delete[] lengths_a;
        delete[] offsets;
        delete[] offsets_a;

        MPI_Finalize();
    }
    catch (string error)
    {
        delete[] a;
        delete[] b;
        delete[] x;
        cout << error;
    }

    return 0;
}