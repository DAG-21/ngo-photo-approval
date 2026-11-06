Rules review, in progress with the owner.

agreed_drafts.json   wording agreed so far, removals and engine changes noted. The banner group has since been APPLIED (profile layered_tuned_v1); anything else in the file is NOT applied.
rules_page.py        regenerates agent-platform/reports/rules_current.html and .json (the 'rules sheet') from the rulebook and the demo database. Run it with the project's Python:
                     image-classifier\.venv\Scripts\python.exe working-notesules_page.py
adopt_tuned_demo.py.history   the one-off script that moved the demo organisation to layered_tuned_v1 (kept as a record; its paths point at the old folders).
rules_page_final.py   writes agent-platform/reports/rules_final.html and .json: the sheet of the profile layered_final_v1 (the owner's review of 2026-11-05), compared with layered_tuned_v1. Reads only the rulebook files and agreed_drafts.json.
                     image-classifier\.venv\Scripts\python.exe working-notes\rules_page_final.py
NOTE: agreed_drafts.json is now APPLIED as the profile layered_final_v1 (not yet adopted by the demo).
