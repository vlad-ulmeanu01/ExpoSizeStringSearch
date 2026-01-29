#pragma once
#include "utils.h"

///struct mostenibil. cu ajutorul lui citim urmatorul string pentru dictionar.
struct DictReader {
    virtual int get_q() = 0;
    virtual std::vector<uint8_t> get_next_t() = 0;
    virtual void reset_iter() = 0;
    virtual void receive_t(const std::vector<uint8_t> t, const std::vector<uint8_t> t_prev, const std::vector<uint8_t> t_foll) = 0;
    virtual std::pair<std::vector<uint8_t>, std::vector<uint8_t>> get_next_neighs() = 0;
};

struct CsesReader: DictReader {
    std::ifstream fin;
    int cnt_strs_read;
    int q;

    CsesReader(std::ifstream fin);

    int get_q();
    std::vector<uint8_t> get_next_t();

    void reset_iter(); ///nefolosit.
    void receive_t(const std::vector<uint8_t> t, const std::vector<uint8_t> t_prev, const std::vector<uint8_t> t_foll); ///nefolosit.
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> get_next_neighs(); ///nefolosit.
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

    void receive_t(const std::vector<uint8_t> t, const std::vector<uint8_t> t_prev, const std::vector<uint8_t> t_foll); ///nefolosit.
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> get_next_neighs(); ///nefolosit.
};

///reader in care pui rezultate dupa ce termina altul.
struct InterReader: DictReader {
    std::vector<std::vector<uint8_t>> ts;

    ///ts tine minte concatentari de gasiri. in chain_neighs tin minte capetele ce nu am putut sa le extind.
    std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> chain_neighs;
    int ind, ind_neighs;

    InterReader();

    int get_q();
    std::vector<uint8_t> get_next_t();
    void reset_iter();
    void receive_t(const std::vector<uint8_t> t, const std::vector<uint8_t> t_prev, const std::vector<uint8_t> t_foll);
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> get_next_neighs();
};
