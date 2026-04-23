f = open('aml_prototype/bank_node/api.py', 'r', encoding='utf-8')
lines = f.readlines()
f.close()

# Line 145 (0-indexed: 144) is: port = BANK_PORTS.get(_BANK_ID, 8001)
# Insert runtime resolution before it
new_lines = lines[:143] + [
    '    _runtime_bank_id = os.environ.get("BANK_ID", "bank_a")\n',
    '    port = BANK_PORTS.get(_runtime_bank_id, 8001)\n',
    '    uvicorn.run(app, host="127.0.0.1", port=port)\n',
]

# Remove old lines 143-146 (the old port+run lines) and keep the rest
# lines[143] = 'if __name__ ...'  <- keep
# lines[144] = '    import uvicorn'  <- keep
# lines[145] = '    from config...'  <- keep
# lines[146] = '    port = BANK_PORTS.get(_BANK_ID, 8001)'  <- replace
# lines[147] = '    uvicorn.run(...)'  <- replace

final_lines = lines[:146] + new_lines[-2:] + lines[148:]

# Actually, let's just rewrite the last block cleanly
final_lines = lines[:143] + [
    'if __name__ == "__main__":\n',
    '    import uvicorn\n',
    '    from config import BANK_PORTS\n',
    '    _runtime_bank_id = os.environ.get("BANK_ID", "bank_a")\n',
    '    port = BANK_PORTS.get(_runtime_bank_id, 8001)\n',
    '    uvicorn.run(app, host="127.0.0.1", port=port)\n',
]

f = open('aml_prototype/bank_node/api.py', 'w', encoding='utf-8')
f.writelines(final_lines)
f.close()
print("OK: patched __main__ block")
