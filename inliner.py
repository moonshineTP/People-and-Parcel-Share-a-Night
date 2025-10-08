"""
    Code taken and modified from https://github.com/dobrakmato/python-import-inliner
    Remember to change output file destination
"""
import sys
import os

sys.path.insert(0, os.getcwd()) 

# This is file to start from.
starting_file = 'oj_final.py'

# This is output (inlined) file.
# CHANGE THIS FOR REAL
output_file = r'C:\Users\admin\Desktop\submission.py'

LOCAL_SOURCE_FOLDERS = ['share_a_ride', 'utils']

# ----------------------------------

modules = []
imports = []

out_file = open(output_file, mode='w', encoding='utf-8')


def count_from_start(string, char):
    c = 0
    for e in string:
        if e == char:
            c += 1
        else:
            return c
    return c


def normalize_import(rel, base='./'):
    dots_from_start = count_from_start(rel, '.')
    down_path = rel[dots_from_start:].split('.')

    base_elements = base.split('/')
    if base_elements[-1] == '':
        base_elements.pop()

    while dots_from_start > 1: # Go up for .. and ...
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
    out_file.write(line)


def inline_file(file, base):
    original_file_path = file

    if not os.path.isfile(file):
        print(f"WARNING: File not found: {original_file_path}!")
        emit_line(f"# ----- file {original_file_path} begin -----\n")
        emit_line(f"# file {original_file_path} does not exist!\n")
        emit_line(f"# ----- file {original_file_path} end -----\n")
        return

    print(f"Inlining file: {file}")

    emit_line(f"\n# ----- file {file} begin -----\n")
    multiline_import = False

    with open(file, encoding='utf-8') as f:
        prev_line = ''
        for l in f:

            if multiline_import:
                emit_line('# ' + l)
                if ')' in l:
                    multiline_import = False
                continue

            stripped_line = l.strip()

            if stripped_line.startswith('import'):
                module = stripped_line.split(' ')[1]

                if ' as ' in stripped_line:
                    as_idx = stripped_line.split(' ').index('as')
                    module = stripped_line.split(' ')[as_idx + 1]

                is_try_except_import = 'try:' in prev_line or 'except ImportError:' in prev_line

                if module not in imports or is_try_except_import:
                    emit_line(l)
                    imports.append(module)
                else:
                    emit_line('# ' + l)

            elif stripped_line.startswith('from'):
                parts = stripped_line.split(' ')
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
                    new_folder = None

                if is_local:
                    if new_file not in modules:
                        modules.append(new_file)
                        inline_file(new_file, new_folder)
                    else:
                        emit_line('# ' + l)

                    if '(' in l and ')' not in l:
                        multiline_import = True
                else: # It's a standard/third-party library import
                    emit_line(l)
            else:
                emit_line(l)
            prev_line = l
    emit_line(f"\n# ----- file {file} end -----\n")


# --- SCRIPT START ---
print("Starting inliner...")

# Check if starting file exists
if not os.path.isfile(starting_file):
    print(f"FATAL ERROR: Starting file '{starting_file}' not found. Please set it at the top of the script.")
else:
    inline_file(starting_file, './')
    out_file.close()
    print(f"Inlining complete. Output file: {output_file}")