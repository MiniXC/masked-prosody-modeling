# run pip-compile, but strip out torch

pip-compile --no-header --no-emit-options --strip-extras --no-annotate --output-file requirements.txt requirements.in
# replace line containing "torch=="
sed -i.backup '/torch==/d' requirements.txt
# remove backup file
rm requirements.txt.backup