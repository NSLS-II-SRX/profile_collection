def update_scanbroker(name, doc):
    if name == "start":
        scanrecord.update_metadata()
        scanrecord.current_scan.put(doc['uid'][:6])
        scanrecord.current_scan_id.put(str(doc['scan_id']))
        scanrecord.scanning.put(True)
        if 'scan' in doc:
            try:
                scanrecord.current_type.put(doc['scan']['type'])
            except:
                pass
    elif name == "stop":
        scanrecord.scanning.put(False)

RE.subscribe(update_scanbroker)
