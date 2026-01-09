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
find code/figures -name '*.txt' -delete
find code/figures -name '*.png' -delete
find code/figures -name '*.svg' -delete

cd ..; zip -r arxiv.zip arxiv