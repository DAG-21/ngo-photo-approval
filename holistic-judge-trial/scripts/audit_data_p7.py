"""My own judgements (Claude looking at each photo): approved photos 242-261 (ids/audit_approved_179.csv, in file order). Client rule text only."""

P7: dict[int, list[tuple[str, str]]] = {
    242: [("R003", "Children's faces hidden behind the books"), ("R004", "Many look into the camera"), ("R005", "Staged hold-up of the books"), ("R011", "Dense, crowded hall")],
    243: [("R002", "Volunteer in a yellow printed shirt and shorts, not white"), ("R003", "Beneficiaries' faces turned away"), ("R011", "Crowded narrow lane behind")],
    244: [("R004", "Everyone looks into the camera"), ("R005", "Posed with folded hands"), ("R006", "No serving shown"), ("R011", "Street and sheds behind")],
    245: [("R004", "Everyone looks into the camera"), ("R005", "Posed with the cheque"), ("R006", "Cheque ceremony; no seva activity"), ("R011", "Cars and houses behind")],
    246: [("R003", "No beneficiary face visible"), ("R007", "Wide angle from behind the volunteer"), ("R008", "Volunteer's back to the camera"), ("R011", "Wet road and parked van")],
    247: [("R007", "Shot from a dark corner with harsh contrast"), ("R009", "Dim and shadowed"), ("R011", "Utensils and clutter around")],
    248: [("R003", "Beneficiary's back to the camera"), ("R011", "Packaged goods clutter the frame")],
    249: [("R004", "Woman looks away or into the camera"), ("R006", "No volunteer and no giving shown"), ("R007", "A car and legs block the view"), ("R011", "Street clutter")],
    250: [("R003", "Patient's face turned and covered by the examination"), ("R011", "Medical clutter on the table")],
    251: [("R004", "Everyone looks into the camera"), ("R005", "Posed line-up with folded hands"), ("R006", "No serving shown")],
    252: [("R004", "Everyone looks into the camera"), ("R005", "Posed group"), ("R006", "No activity shown"), ("R010", "A phone screenshot with a timestamp overlay")],
    253: [("R004", "Beneficiary looks straight into the camera"), ("R005", "Posed with the tray and child"), ("R011", "Crowd and boards behind")],
    254: [("R004", "Girls look into the camera"), ("R005", "Posed on a tricycle"), ("R006", "No volunteer and no giving shown"), ("R011", "Rubbish on the road")],
    255: [("R011", "Open, littered ground")],
    256: [("R002", "Volunteer in a cream polo, not white"), ("R011", "Posts and sheds behind")],
    257: [("R003", "Children's faces turned away"), ("R011", "Wall and windows behind")],
    258: [("R001", "Banner text partly hidden by the volunteer's head"), ("R004", "Boy looks into the camera")],
    259: [("R004", "Everyone looks into the camera"), ("R005", "Posed line-up"), ("R006", "No serving; the food is cut off at the bottom"), ("R011", "Cart and rubbish")],
    260: [("R003", "Face half covered by a scarf"), ("R004", "Looks straight into the camera"), ("R006", "No volunteer and no giving shown")],
    261: [("R004", "Looks into the camera"), ("R006", "No volunteer and no giving shown"), ("R011", "Hospital seating clutter")],
}
