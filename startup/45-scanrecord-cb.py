def update_scanbroker(name, doc):
    try:
        if name == "start":
            scanrecord.update_metadata()
            scanrecord.current_scan.set(doc['uid'][:6], timeout=5)
            scanrecord.current_scan_id.set(str(doc['scan_id']), timeout=5)
            scanrecord.scanning.set(True, timeout=5)
            if 'scan' in doc:
                try:
                    scanrecord.current_type.set(doc['scan']['type'], timeout=5)
                except:
                    pass
        elif name == "stop":
            scanrecord.scanning.set(False, timeout=5)
    except WaitTimeoutError as e:
        print('Timeout error! Continuing...')

RE.subscribe(update_scanbroker)
