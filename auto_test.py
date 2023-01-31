import os
import math
from subprocess import PIPE, Popen

RELATIVE_TOLERANCE = 1e-6
ZERO_ISH = 1e-50


def get_sample_output(test):
    return open(
        f"auto_test_outputs/"
        + "_".join([f"{k}_{v}" for (k, v) in sorted([(k, v) for k, v in test.items()])])
        + ".cor"
    ).read()


def run_code(version, test):
    cmd = f"./auto_build/kernel_{version} " + " ".join(
        [f"-{k} {v}" for k, v in test.items()]
    )
    proc = Popen(
        cmd,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
    )
    run_out, run_err = proc.communicate()
    return run_out.decode(), run_err.decode(), proc.returncode


def compare_outputs(output, correct, n):
    out_lines = output.split("\n")
    cor_lines = correct.split("\n")
    if len(out_lines) < len(cor_lines):
        return "Not enough lines"

    for i, (out, cor) in enumerate(list(zip(out_lines, cor_lines))[12 : 12 + n]):
        try:
            diff = any(
                math.isnan(float(o)) or abs(float(o)) > ZERO_ISH
                if abs(float(c)) < ZERO_ISH
                else abs((float(o) - float(c)) / float(c)) > RELATIVE_TOLERANCE
                for o, c in [(out.split()[i], cor.split()[i]) for i in [3, 5, 7]]
            )
        except:
            diff = True
        if diff:
            return (
                f"First difference on line {i}\n"
                + f"Output:  {out.strip()}\n"
                + f"Correct: {cor.strip()}\n"
                + (
                    f"Next out: {out_lines[13+i].strip()}\nNext cor: {cor_lines[13+i].strip()}"
                    if i < n - 1
                    else ""
                )
            )

    return None


def main():
    print("=== Compiling ===")
    make_out, make_err = Popen(
        "mkdir -p auto_build && cd auto_build && rm -rf * && cmake .. && make -j 24",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
    ).communicate()

    err_string = f"STDOUT:\n{make_out.decode()}\nSTDERR:\n{make_err.decode()}"

    if "error" in str(make_err) or "CMake Error" in str(make_err):
        print("Error while running 'make':")
        print(err_string)

    with open("auto_test_outputs/make.out", "w") as f:
        f.write(err_string)

    warnings = (str(make_out) + str(make_err)).count("warning")
    if warnings:
        print(f"Make ran with ~{warnings} warnings")

    tests = [
        {
            "n": 20,
            "s": 1,
            "m": 1,
            "f": 10,
            "d": 2,
            "g": 0.981,
            "b": 3,
            "o": 0,
            "t": 0.05,
            "i": 400,
        },
        {
            "n": 50,
            "s": 0.5,
            "m": 2,
            "f": 5,
            "d": 4,
            "g": 2,
            "b": 5,
            "o": 3,
            "t": 0.02,
            "i": 400,
        },
        {
            "n": 200,
            "s": 0.08,
            "m": 0.5,
            "f": 10,
            "d": 3,
            "g": 2,
            "b": 5,
            "o": 0,
            "t": 0.02,
            "i": 400,
        },
    ]

    for imp in ["main", "opt", "sse", "vect_omp", "omp"]:
        print(f"\n=== Testing {imp} implementation ===")
        for test in tests:
            correct_output = get_sample_output(test)
            if imp == "omp":
                test["p"] = 24
            stdout, stderr, err_code = run_code(imp, test)
            if err_code == 127:
                print(f"No {imp} implementation found.")
                break
            with open(
                f"auto_test_outputs/{imp}_"
                + "_".join(
                    [f"{k}_{v}" for (k, v) in sorted([(k, v) for k, v in test.items()])]
                )
                + ".out",
                "w",
            ) as out_file:
                out_file.write(stdout + stderr)
            if err_code:
                print(f"Error running test {test}")
                if err_code == -11:
                    print("Segmentation fault.")
                print(stderr)
                break

            comparison = compare_outputs(stdout, correct_output, n=400)
            if comparison:
                print(f"Difference on test {test}")
                print(comparison)
                break
        else:
            print(f"{imp} looks good to me!")


if __name__ == "__main__":
    main()
