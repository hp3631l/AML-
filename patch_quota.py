f = open('aml_prototype/simulator/scenarios.py', 'r', encoding='utf-8')
lines = f.readlines()
f.close()

# Lines 129-143 (0-indexed: 128-142) are the quota dict
# Replace them
new_lines = lines[:128] + [
    '    quota = {\n',
    '        "hybrid_slow_cross":  10,     # slow cross-country chain (hybrid + low_and_slow + cross_country)\n',
    '        "hybrid_fanout_peel":  6,     # fan-out + peel-off (hybrid + can be slow)\n',
    '        "hybrid_sg_loop":      4,     # scatter-gather + loop (hybrid + can be slow)\n',
    '        "base_chain_slow":    16,     # chain, slow variant (low_and_slow)\n',
    '        "base_loop_slow":     12,     # recursive loop, slow (low_and_slow)\n',
    '        "base_sg_cross":      10,     # scatter-gather, cross-country\n',
    '        "base_peel":           8,     # peel-off\n',
    '        "base_fan_in":         8,     # fan-in\n',
    '        "base_fan_out":        8,     # fan-out\n',
    '        "base_burst":          6,     # burst\n',
    '        "base_chain_fast":     6,     # chain, fast variant\n',
    '        "base_bot":            6,     # agentic bot\n',
    '    }\n',
    '    # Total: 100\n',
] + lines[143:]

f = open('aml_prototype/simulator/scenarios.py', 'w', encoding='utf-8')
f.writelines(new_lines)
f.close()
print("OK: quota patched")
