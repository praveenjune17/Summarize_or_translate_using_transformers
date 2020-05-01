import os
import difflib
from scripts.creates import log
from scripts.configuration import *

to_be_flaked_directory_path = input("Enter the path of the modified scripts directory :")#'C:/Users/Vinodhkumar/Summarize_or_translate_using_transformers/scripts/'
existing_cloned_directory_path = input("Enter the path of the unchanged scripts directory :")#'C:/Users/Vinodhkumar/Summarize_or_translate_using_transformers - Copy/scripts'
os.chdir(to_be_flaked_directory_path)

for file in os.listdir(to_be_flaked_directory_path):
    assert not os.system(f'autoflake --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys {file}'), f'Issue while running autoflake on {file}'
log.info('Autoflake run completed')

assert len(os.listdir(to_be_flaked_directory_path)) == len(os.listdir(existing_cloned_directory_path)), 'Number of files are not matching'
# Check for duplicate keys in the config file
assert len(model_parms.keys()) + len(training_parms.keys()) + len(token_ids.keys()) + len(file_path.keys()) + len(h_parms.keys()) == len(config.keys()), 'similar key values between the differe added dictonaries'

# Diff checker
log.info('''Please go through the modifications made by autoflake script in the diff check log,
            especially focus on the '+' (added lines) ''')
for flaked_file, existing_file in zip(sorted(os.listdir(to_be_flaked_directory_path)), sorted(os.listdir(existing_cloned_directory_path))):
    with open(os.path.join(to_be_flaked_directory_path, flaked_file), 'r') as flaked_file_obj:
        flaked_file_text = flaked_file_obj.read().splitlines()
    with open(os.path.join(existing_cloned_directory_path, existing_file), 'r') as existing_file_obj:
        existing_file_text = existing_file_obj.read().splitlines()
    for line in difflib.unified_diff(existing_file_text, 
                                     flaked_file_text, 
                                     tofile=os.path.join(to_be_flaked_directory_path, flaked_file), 
                                     fromfile=os.path.join(existing_cloned_directory_path,existing_file)):
        log.info(line)