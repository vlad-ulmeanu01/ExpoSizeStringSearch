#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>
#include <arrow/pretty_print.h>
#include <arrow/type.h>

#include <iostream>
#include <vector>
#include <cstdint>

void read_single_column() {
    std::cout << "Reading first column of parquet-arrow-example.parquet" << std::endl;
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile,
                            arrow::io::ReadableFile::Open("/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/the_pile_10token_strings_cached/train-split-00000.parquet",
                                                          arrow::default_memory_pool()));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::ChunkedArray> array;
    PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));

    std::cout << "array->num_chunks(): " << array->num_chunks() << '\n';
    std::cout << "array col size: " << array->chunk(0)->length() << '\n';

    std::shared_ptr<arrow::StringArray> sa = std::static_pointer_cast<arrow::StringArray>(array->chunk(0));

    ///iterates through all strings and prints those that contain a 0 midway through.
    for (int i = 0; i < sa->length(); i++) {
        //std::string buf = sa->GetString(i);
        //std::vector<uint8_t> v(buf.data(), buf.data() + buf.size());
            
        //std::vector<uint8_t> v(sa->value_data()->data() + sa->value_offset(i), sa->value_data()->data() + (i+1 < sa->length()? sa->value_offset(i+1): sa->value_data()->size()));
        std::vector<uint8_t> v(sa->value_data()->data() + sa->value_offset(i), sa->value_data()->data() + sa->value_offset(i+1));

        if (*std::min_element(v.begin(), v.end())) continue;

        std::cout << "i, size = " << i << ' ' <<  v.size() << ": ";
        for (uint8_t x: v) std::cout << x << ' ';
        std::cout << '\n';
    }

    //PARQUET_THROW_NOT_OK(arrow::PrettyPrint(*array, 4, &std::cout));
    //std::cout << std::endl;
}

int main(int argc, char** argv) {
    read_single_column();
}
