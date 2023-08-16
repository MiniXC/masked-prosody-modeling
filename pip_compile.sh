# run pip-compile, but strip out requirements not in requirements.in

# first, sort requirements.in
sort -o requirements.in requirements.in

pip-compile --no-header --no-emit-options --strip-extras --no-annotate --output-file requirements.txt requirements.in
# for each line in requirements.txt, check if it's in requirements.in (split by == first)
# if it is, then keep it, otherwise remove it
for line in $(cat requirements.txt); do
    line_old=$line
    line=$(echo $line | cut -d'=' -f1)
    if grep -qx $line requirements.in; then
        echo $line_old >> requirements.tmp
    fi
done
# remove lines not containing ==
sed -i.backup '/=/!d' requirements.tmp
rm requirements.tmp.backup
rm requirements.txt
mv requirements.tmp requirements.txt