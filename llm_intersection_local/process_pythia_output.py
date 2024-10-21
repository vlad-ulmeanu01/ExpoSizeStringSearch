import os

def main():
    cast_ch = lambda ch: ch if ord(ch) < 256 else chr(255)
    cast_string = lambda s: ''.join(map(cast_ch, [ch for ch in s]))

    fpath = "/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/outputs_pythia_run1"
    with open(f"{fpath}/outputs_concat.txt", "w") as fout:
        for entry in os.scandir(fpath):
            if entry.is_file() and not entry.name.endswith("concat.txt"):
                # print(entry.name, entry.path)
                with open(entry.path) as fin:
                    for line in fin.readlines():
                        fout.write(cast_string(line))
                fout.write('\n')
                        

if __name__ == "__main__":
    main()