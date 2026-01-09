rm -rf ../arxiv
mkdir -p ../arxiv
cp -r * ../arxiv/

cd ../arxiv
rm -f code/*.*
rm -rf code/data
rm -rf drafts
rm -rf feedback
rm -rf misc
rm -f arxiv.sh
rm -f arxiv.zip
rm -f AGUJournalTemplate.tex
rm -f si_template_2019.tex
rm -f agusample.bib
rm -f agutexSI2019.cls

find . -name '.DS_Store' -delete
find . -name '.gitignore' -delete

# Extract referenced files from main.tex
referenced_files=$(grep -oE '\{code/[^}]+\}' main.tex | tr -d '{}' | sort -u)

# Delete files in code/ that are not referenced in main.tex
find code -type f | while read file; do
  if ! echo "$referenced_files" | grep -q "^${file}$"; then
    rm -f "$file"
  fi
done

# Delete empty directories in code/
find code/ -type d -empty -delete

cd ..; zip -r arxiv.zip arxiv