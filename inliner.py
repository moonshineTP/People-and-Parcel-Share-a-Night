"""
    Code taken and modified from https://github.com/dobrakmato/python-import-inliner
    Remember you can redirect stdout to a file, e.g.:
        pypy3 -m inliner > submission
        pypy3 -m inliner oj.py > submission
"""
import sys
import os

sys.path.insert(0, os.getcwd())

# This is file to start from (can be overridden by argv[1]).
STARTING_FILE = 'oj.py'

# Local source folders considered for inlining
LOCAL_SOURCE_FOLDERS = ['share_a_ride']

# Container for tracked modules and imports
modules = []
imports = []

# Output file (stdout by default)
out_file = sys.stdout


def count_from_start(string, char):
    """
    Count occurrences of char from the start of string.
    """
    c = 0
    for e in string:
        if e == char:
            c += 1
        else:
            return c
    return c


def normalize_import(rel, base='./'):
    """
    Normalize relative import to absolute file path.
    """
    dots_from_start = count_from_start(rel, '.')
    down_path = rel[dots_from_start:].split('.')

    base_elements = base.split('/')
    if base_elements[-1] == '':
        base_elements.pop()

    while dots_from_start > 1:  # Go up for .. and ...
        if base_elements:
            base_elements.pop()
        dots_from_start -= 1

    final = base_elements + down_path
    file = '/'.join(final) + '.py'

    if final:
        final.pop()
    folder = '/'.join(final)
    if not folder:
        folder = './'
    elif not folder.endswith('/'):
        folder += '/'

    return file, folder


def emit_line(line):
    """
    Emit line to output file.
    """
    out_file.write(line)


def inline_file(file, base):
    """
    Main recursive inlining function.
    """
    original_file_path = file

    # Resolve relative imports
    if not os.path.isfile(file):
        print(f"WARNING: File not found: {original_file_path}!", file=sys.stderr)
        emit_line(f"# ----- file {original_file_path} begin -----\n")
        emit_line(f"# file {original_file_path} does not exist!\n")
        emit_line(f"# ----- file {original_file_path} end -----\n")
        return

    print(f"Inlining file: {file}", file=sys.stderr)

    # Emit file begin marker
    emit_line(f"\n# ----- file {file} begin -----\n")

    # Collect next imports to process
    multiline_import = False    # Track multi-line imports
    with open(file, encoding='utf-8') as f:
        prev_line = ''
        for l in f:
            # Handle multi-line imports
            if multiline_import:
                # Comment out continued import lines in the output
                emit_line('# ' + l)
                if ')' in l:
                    multiline_import = False
                prev_line = l
                continue

            # Process normal lines
            stripped_line = l.strip()
            if stripped_line.startswith('import'):
                module = stripped_line.split(' ')[1]

                # Handle "import X as Y"
                if ' as ' in stripped_line:
                    as_idx = stripped_line.split(' ').index('as')
                    module = stripped_line.split(' ')[as_idx + 1]
                is_try_except_import = 'try:' in prev_line or 'except ImportError:' in prev_line

                # Check if it's a local import (in our source folders)
                if module not in imports or is_try_except_import:
                    emit_line(l)
                    imports.append(module)
                else:   # It's already imported, comment it out
                    emit_line('# ' + l)

            elif stripped_line.startswith('from'):
                parts = stripped_line.split(' ')

                # Check if we have enough parts for a valid from import
                if len(parts) < 2:
                    emit_line(l)
                    prev_line = l
                    continue

                module_path = parts[1]
                top_level_module = module_path.split('.')[0]

                # Check if it's a local import (relative OR in our source folders)
                is_local = False
                if module_path.startswith('.'):
                    is_local = True
                    new_file, new_folder = normalize_import(module_path, base)
                elif top_level_module in LOCAL_SOURCE_FOLDERS:
                    is_local = True
                    new_file = module_path.replace('.', '/') + '.py'
                    new_folder = '/'.join(new_file.split('/')[:-1]) + '/'
                else:
                    new_file = None
                    new_folder = None

                # Inline local imports
                if is_local:
                    if new_file not in modules:
                        modules.append(new_file)
                        inline_file(new_file, new_folder)
                    else:
                        emit_line('# ' + l)

                    if '(' in l and ')' not in l:
                        multiline_import = True
                else:  # Third party or standard library import
                    emit_line(l)
            # Normal line, just emit
            else:
                emit_line(l)

            # Track previous line for try-except import detection
            prev_line = l

    # Emit file end marker
    emit_line(f"\n# ----- file {file} end -----\n")




# --- SCRIPT START ---
def _main():
    print("Starting inliner...", file=sys.stderr)

    # Allow overriding starting file via CLI: `pypy3 -m inliner my_start.py > submission`
    global STARTING_FILE        # pylint: disable=global-statement
    if len(sys.argv) >= 2 and sys.argv[1]:
        STARTING_FILE = sys.argv[1]

    # Check if starting file exists
    if not os.path.isfile(STARTING_FILE):
        print(
            f"FATAL ERROR: Starting file '{STARTING_FILE}' not found.\n"
            f"Please set it or pass it as argv[1].", file=sys.stderr
        )
        return 2
    else:
        inline_file(STARTING_FILE, './')
        out_file.flush()
        print("Inlining complete.", file=sys.stderr)
        return 0




if __name__ == "__main__":
    sys.exit(_main())
