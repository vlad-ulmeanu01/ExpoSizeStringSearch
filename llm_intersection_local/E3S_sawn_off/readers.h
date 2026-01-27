#pragma once
#include "utils.h"

///struct mostenibil. cu ajutorul lui citim urmatorul string pentru dictionar.
struct DictReader {
    virtual int get_q() = 0;
    virtual std::vector<uint8_t> get_next_t() = 0;
};

struct CsesReader: DictReader {
    std::ifstream fin;
    int cnt_strs_read;
    int q;

    CsesReader(std::ifstream fin);

    int get_q();

    std::vector<uint8_t> get_next_t();
};

struct ParquetChunkReader: DictReader {
    int cnt_parquet_files;
    int pq_ind, file_ind, row_ind, col_ind;
    bool have_next;
    int q;

    std::shared_ptr<arrow::io::ReadableFile> infile;
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::ChunkedArray> array;
    std::shared_ptr<arrow::StringArray> sa;

    ParquetChunkReader();

    int get_q();

    void setup_parquet_file(int ind);

    std::vector<uint8_t> get_next_t();
};

///TODO mai trebuie un reader care pune rezultate dupa ce termina primul.
