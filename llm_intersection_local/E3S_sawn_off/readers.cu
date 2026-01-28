#include "readers.h"
#include "utils.h"


CsesReader::CsesReader(std::ifstream fin): fin(std::move(fin)), cnt_strs_read(0) {
    this->fin >> q; ///!!
}

int CsesReader::get_q() {
    return q;
}

///intoarce true <=> mai exista inca un string de citit.
std::vector<uint8_t> CsesReader::get_next_t() {
    if (cnt_strs_read >= q) return std::vector<uint8_t>();
    std::string t; fin >> t;
    std::vector<uint8_t> vt(t.begin(), t.end());
    cnt_strs_read++;
    return vt;
}

void CsesReader::receive_t(const std::vector<uint8_t>& t) { assert(false); }

void CsesReader::reset_iter() { assert(false); }


ParquetChunkReader::ParquetChunkReader(int cnt_parquet_files): cnt_parquet_files(cnt_parquet_files), pq_ind(0), file_ind(0), row_ind(0), col_ind(0), have_next(true), q(0) {
    for (int i = 0; i < cnt_parquet_files; i++) {
        setup_parquet_file(i);
        for (int z = 0; z < sa->length(); z++) q += (sa->value_offset(z+1) - sa->value_offset(z) + PARQUET_BYTES_PER_READ - 1) / PARQUET_BYTES_PER_READ;
    }

    setup_parquet_file(0);
}

void ParquetChunkReader::reset_iter() {
    pq_ind = 0;
    file_ind = 0;
    row_ind = 0;
    col_ind = 0;
    have_next = true;
    setup_parquet_file(0);
}

int ParquetChunkReader::get_q() {
    return q;
}

void ParquetChunkReader::setup_parquet_file(int ind) {
    PARQUET_ASSIGN_OR_THROW(infile,
        arrow::io::ReadableFile::Open(
            PARQUET_DIR + "train-" + pad_parquet_fname(ind) + "-of-01650.parquet",
            arrow::default_memory_pool()
        )
    );

    PARQUET_ASSIGN_OR_THROW(reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));    
    PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));

    sa = std::static_pointer_cast<arrow::StringArray>(array->chunk(0));
}

std::vector<uint8_t> ParquetChunkReader::get_next_t() {
    if (!have_next) return std::vector<uint8_t>();
    have_next = (
        pq_ind + 1 < cnt_parquet_files ||
        row_ind + 1 < sa->length() ||
        sa->value_offset(row_ind) + col_ind + PARQUET_BYTES_PER_READ < sa->value_offset(row_ind+1)
    );

    std::vector<uint8_t> vt(
        sa->value_data()->data() + sa->value_offset(row_ind) + col_ind,
        sa->value_data()->data() + std::min(sa->value_offset(row_ind+1), sa->value_offset(row_ind) + col_ind + PARQUET_BYTES_PER_READ)
    );

    col_ind += vt.size();
    if (sa->value_data()->data() + sa->value_offset(row_ind) + col_ind >= sa->value_data()->data() + sa->value_offset(row_ind+1)) {
        col_ind = 0;
        row_ind++;
        if (row_ind >= sa->length()) {
            row_ind = 0;
            pq_ind++;
            if (pq_ind < cnt_parquet_files) setup_parquet_file(pq_ind);
        }
    }

    return vt;
}

void ParquetChunkReader::receive_t(const std::vector<uint8_t>& t) { assert(false); }


InterReader::InterReader(): ind(0) {}

int InterReader::get_q() {
    return ts.size();
}

void InterReader::receive_t(const std::vector<uint8_t>& t) {
    ts.push_back(t);
}

std::vector<uint8_t> InterReader::get_next_t() {
    if (ind >= (int)ts.size()) return std::vector<uint8_t>();
    return ts[ind++];
}

void InterReader::reset_iter() {
    ind = 0;
}
