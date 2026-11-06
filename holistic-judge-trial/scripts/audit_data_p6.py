"""My own judgements (Claude looking at each photo): approved photos 222-241 (ids/audit_approved_179.csv, in file order). Client rule text only."""

P6: dict[int, list[tuple[str, str]]] = {
    222: [("R004", "Children look away or at the camera; none serving"), ("R006", "No volunteer and no giving shown"), ("R011", "Rubbish on the ground and a cart behind")],
    223: [("R003", "Faces not visible; queue seen from the side and behind"), ("R006", "Shot from far; serving too small to see"), ("R007", "Wide angle from the side of the queue")],
    224: [("R003", "Face turned down"), ("R006", "No volunteer and no giving shown")],
    225: [("R003", "No beneficiary face visible"), ("R008", "Volunteers' backs to the camera"), ("R011", "Street and building behind")],
    226: [("R003", "Beneficiary's back to the camera"), ("R011", "Boxes and signboards behind")],
    227: [("R003", "Children's backs to the camera"), ("R011", "Shelves and boards behind")],
    228: [("R006", "Only birds; no volunteer or feeding action"), ("R011", "Rubbish and tarps on the ground")],
    229: [("R004", "Volunteers look into the camera"), ("R005", "Posed line-up behind the food"), ("R006", "No beneficiary and no serving in progress")],
    230: [("R004", "Beneficiary looks straight into the camera"), ("R005", "Posed with the tray")],
    231: [("R004", "Everyone looks into the camera"), ("R005", "Posed group"), ("R006", "No beneficiary and no activity"), ("R011", "Cluttered shed behind")],
    232: [("R001", "Banner text partly hidden by the people")],
    233: [("R003", "Children's faces turned away"), ("R011", "Wet, cluttered ground and shed behind")],
    234: [("R004", "Beneficiary looks into the camera"), ("R005", "Posed handover")],
    235: [("R006", "Only a truck with a banner; no people or activity")],
    236: [("R003", "Beneficiary's face hidden under the hat"), ("R011", "Bricks, poles and signs crowd the frame")],
    237: [("R006", "Two men talking under an umbrella; no seva activity")],
    238: [("R003", "Children's backs to the camera"), ("R001", "Banner partly hidden by the people")],
    239: [("R004", "Everyone looks into the camera"), ("R005", "Posed line-up"), ("R006", "No beneficiary and no serving in progress")],
    240: [("R003", "Faces not visible; queue seen from far away"), ("R006", "Shot from far; serving too small to see")],
    241: [("R003", "Beneficiary's face turned away"), ("R011", "Boxes and shed behind")],
}
