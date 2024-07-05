import pickle
import os, re

def extract_string_between_underscore_and_dot(input_string):
    # pattern = r'_(.*?)\.'  # The regex pattern to match the text between '_' and '.'
    pattern = r'_(\d*?)\.'  # The regex pattern to match the digits between '_' and '.'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)  # Return the matched substring between '_' and '.'
    else:
        return None  # Return None if no match is found

def get_number_from_fname(s):
    # This function extracts the numerical value from the string
    # You can customize this function based on your specific requirements
    try:
        return int(extract_string_between_underscore_and_dot(s))
    except ValueError:
        return float('inf')  # Return a large value if the string doesn't contain a valid number

def get_files_with_prefix(folder_path, prefix):
    files_with_prefix = [f for f in os.listdir(folder_path) if f.startswith(prefix) and os.path.isfile(os.path.join(folder_path, f))]

    files_with_prefix = []
    for f in os.listdir(folder_path):
        if f.startswith(prefix):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path):
                files_with_prefix.append(full_path)
    return files_with_prefix

def get_rest_idx(args, num_groups, folder_path, prefix):
    '''return a list pf all the idx that are not generated,
    generate the rest to form a whole dataset'''

    pkl_list = get_files_with_prefix(folder_path, prefix)
    pkl_list = sorted(pkl_list, key=get_number_from_fname) ## must sort
    ## sanity check if subdivision matched
    with open(pkl_list[0], 'rb') as f:
        d = pickle.load(f)
        assert num_groups == len(d['observations']) * args.num_parts
    gen_idx = list(map(get_number_from_fname, pkl_list))

    rest_idx = []
    ## checkdesign slow, could be improved
    for i in range(args.num_parts):
        if i not in gen_idx:
            rest_idx.append(i)
    return rest_idx

def get_notexist_savepath(filename):
    savename = filename
    cnt = 1
    # Check if the file exists
    while os.path.exists(savename):
        # Split the filename and extension
        base, ext = os.path.splitext(filename)
        suffix = f'_{cnt}'
        # Append the suffix to the filename
        savename = f"{base}{suffix}{ext}"
        cnt += 1

    return savename
