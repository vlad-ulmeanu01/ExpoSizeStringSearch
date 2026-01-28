#pragma once
#include "utils.h"

///struct mostenibil. cu ajutorul lui citim urmatorul string pentru dictionar.
struct DictReader {
    virtual int get_q() = 0;
    virtual std::vector<uint8_t> get_next_t() = 0;
    virtual void receive_t(const std::vector<uint8_t>& t) = 0;
    virtual void reset_iter() = 0;
};

struct CsesReader: DictReader {
    std::ifstream fin;
    int cnt_strs_read;
    int q;

    CsesReader(std::ifstream fin);

    int get_q();
    std::vector<uint8_t> get_next_t();

    void receive_t(const std::vector<uint8_t>& t); ///nefolosit.
    void reset_iter(); ///nefolosit.
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

    ParquetChunkReader(int cnt_parquet_files);

    int get_q();
    std::vector<uint8_t> get_next_t();
    void setup_parquet_file(int ind); ///e doar apelat din clasa, deci nu ai nevoie de definire in DictReader.
    void reset_iter();

    void receive_t(const std::vector<uint8_t>& t); ///nefolosit.
};

///reader in care pui rezultate dupa ce termina altul.
struct InterReader: DictReader {
    std::vector<std::vector<uint8_t>> ts;
    int ind;

    InterReader();

    int get_q();
    std::vector<uint8_t> get_next_t();
    void receive_t(const std::vector<uint8_t>& t);
    void reset_iter();
};
