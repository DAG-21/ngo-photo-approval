# Review desk prototype

The Claude Design canvas "Photo Review Desk redesign" built as a working page: activities list, review grid (AI verdict on every photo), one photo full size, and the phone layout (same page, narrow screen).
Plain HTML, CSS and JavaScript. It does not use the AYSG monorepo. Nothing is saved or sent: decisions live in the page and are gone on reload.

Run it (from this folder):

    python -m http.server 8200      # then open http://127.0.0.1:8200/

The photos and verdicts are NOT in git (`photos/` and `data/` are ignored: they are the NGO's photographs). Make them with

    cd ../agent-platform
    python scripts/review_desk_data.py                 # dry run: lists the photos and the cost, no model call
    python scripts/review_desk_data.py --from-demo     # only photos the local demo has already judged, no model call
    python scripts/review_desk_data.py --go            # about 50 photos in total; Claude Haiku 5.5 judges the rest (photos go to Anthropic, about Rs 8)

The staff's own decision for each photo is kept out of the page (reports/review_desk_staff.json in agent-platform) and is used only to check the AI.
