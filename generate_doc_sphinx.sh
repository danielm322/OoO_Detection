
if [ -d "docs" ]
then
    rm -rf docs
fi

pip3 install -U sphinx
pip3 install sphinx-rtd-theme
pip3 install myst-parser
pip3 install sphinx-design
pip3 install sphinx-copybutton

mkdir docs
cd docs
sphinx-quickstart \
    --quiet \
    --project="OOD Detection using DNN Latent Representations Uncertainty" \
    --author="CEA-LSEA" \
    --release="1.0.0-rc" \
    --language=en \
    --ext-autodoc \
    --ext-todo \
    --ext-ifconfig \
    --ext-mathjax \
    --ext-viewcode \
    --extensions=sphinx.ext.napoleon,sphinx_rtd_theme,myst_parser,sphinx_design,sphinx_copybutton
sed -i 's/alabaster/sphinx_rtd_theme/g' conf.py
# Useless ?
sed -i 's/#project-information/\n\nimport os\nimport sys\n\nsys.path.insert(0, os.path.abspath(".."))\n\n#project-information/' conf.py

sed -i '/^Indices and tables/i \\   README\n   modules\n' index.rst

cd ..
pip install -r requirements.txt
pip3 install .

sphinx-apidoc -o docs ls_ood_detect_cea

cp README.md docs/

mkdir docs/ls_ood_detect_cea
cp ls_ood_detect_cea/CEA-LSEA-OoD_Detection_DNN_Latent_Space.* docs/ls_ood_detect_cea/
mkdir docs/ls_ood_detect_cea/images
cp ls_ood_detect_cea/images/* docs/ls_ood_detect_cea/images/
cp identity_card.yml docs/
cd docs
make html
# Copy image for html from readme which isn't rendered by sphinx
mkdir _build/html/assets
cp ../assets/Logo_ConfianceAI.png _build/html/assets/