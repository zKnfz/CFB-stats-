"""Inject data/site_data.json into src/template.html -> gridiron-t-rank.html"""
import os
from paths import DATA_DIR
here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "template.html")).read()
d = open(os.path.join(DATA_DIR, "site_data.json")).read().replace("</", "<\\/")
out = os.path.join(here, "..", "gridiron-t-rank.html")
open(out, "w").write(t.replace("__DATA__", d)); print("wrote", os.path.abspath(out))
