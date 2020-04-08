#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"
#include <algorithm>

using namespace std;

void generate(int size)
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

int main(int argc, char* argv[])
{
    try
    {
        MPI_Init(&argc, &argv);
        int size, rank;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int m = 300;            // <---- put here matrix size 
        int n;

        int size_m = size;
        if (m % size == 0) 
            size_m = m;
        else if (m > size) 
            size_m = m + size - (m % size);
        
        double eps = 0.001;
        long double* a, * b, * x;

        a = new long double [m * size_m];
        b = new long double [size_m];
        x = new long double [size_m];

        if (rank == 0)
        {
            //generate(m);

            string name = "matrix" + to_string(m) + ".txt";
            cout << "Reading file " << name << endl;
            ifstream in;

            in.open(name);
            if (!in.is_open())
                throw "Can't open file " + name;

            in >> m >> n;
            if (n != m + 1)
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


            cout << "Checking matrix " << endl;
            for (auto i = 0; i < m; i++)
            {
                double sum = 0;
                for (auto j = 0; j < m; j++)
                    if (i != j)
                        sum += a[i * m + j];
                if (a[i * m + i] <= sum || a[i * m + i] == 0)
                    throw "Can't work with this matrix! Row " + to_string(i + 1) + " is bad";
            }
            cout << "Matrix is ok" << endl;
        }
        
        int length = size_m / size;
        long double* temp_a = new long double [m * length];
        long double* temp_b = new long double [length];        
        long double* temp_x = new long double [length];
        long double* norms = new long double [size];

        int* lengths = new int[size];
        for (int i = 0; i < size; i++)
            lengths[i] = length;
        lengths[size - 1] += m - size_m;        // real size of the last block
        if (size > m)
        {
            for (int i = m; i < size; i++)
                lengths[i] = 0;
        }


        MPI_Scatter(a, length * m, MPI_LONG_DOUBLE, temp_a, length * m, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, length, MPI_LONG_DOUBLE, temp_b, length, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(x, length, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(norms, size, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

        // Jakobi
        long double norm = 1;
        const auto start = MPI_Wtime();
        while (norm > eps)
        {
            for (auto i = 0; i < lengths[rank]; i++)
            {
                temp_x[i] = temp_b[i];
                for (auto j = 0; j < m; j++)
                {
                    if (i + length * rank != j)
                        temp_x[i] -= temp_a[i * m + j] * x[j];
                }
                temp_x[i] /= temp_a[i * m + i + length * rank];
            }

            // find max norm
            norm = 0;
            for (auto i = 0; i < lengths[rank]; i++)
            {
                if (abs(x[rank * length + i] - temp_x[i]) > norm)
                    norm = abs(x[rank * length + i] - temp_x[i]);
                x[rank * length + i] = temp_x[i];
            }

            MPI_Allgather(&norm, 1, MPI_LONG_DOUBLE, norms, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
            //MPI_Reduce(&norm, &res_norm, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Allgather(temp_x, length, MPI_LONG_DOUBLE, x, length, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

            norm = norms[0];
            for (int i = 0; i < size; i++)
            {
                if (norms[i] > norm)
                    norm = norms[i];
            }
        }
        const auto finish = MPI_Wtime();
        delete[] temp_x; 

        if (rank == 0)
        {
            cout << "Number of processes: " << size << " Size: " << m << " Time: " << finish - start << endl;
            ofstream out;
            out.open("result.txt");

            out << m << endl;
            for (auto i = 0; i < m; i++)
                out << x[i] << endl;
            out.close();
        }

        delete[] a;
        delete[] temp_a;
        delete[] b;
        delete[] temp_b;
        delete[] x;
        delete[] norms;
        delete[] lengths;

        MPI_Finalize();
    }
    catch (exception e)
    {
        cout << e.what();
    }

    return 0;
}