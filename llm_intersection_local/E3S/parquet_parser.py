import pyarrow.parquet as pq
import sys

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python parquet_parser.py <parquet_fpath>.")
        return

    fpath = sys.argv[1]
    table = pq.read_table(fpath)
    for s in table[0]:
        s = str(s).encode("utf-8")
        if len(s) > 1:
            sys.stdout.buffer.write(s)
            sys.stdout.flush()
            sys.stdin.readline()

    sys.stdout.write('\n') # TODO poate fixezi asta.
    sys.stdout.flush()

if __name__ == "__main__":
    main()
