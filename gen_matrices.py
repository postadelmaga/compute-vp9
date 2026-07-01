import math

def get_matrix_val(i, j, size):
    if size == 4:
        c0, c1, angle_scale = 0.5, 0.70710678, 0.39269908
    elif size == 8:
        c0, c1, angle_scale = 0.35355339, 0.5, 0.19634954
    elif size == 16:
        c0, c1, angle_scale = 0.25, 0.35355339, 0.09817477
    else:
        c0, c1, angle_scale = 0.17677669, 0.25, 0.049087385
        
    if i == 0:
        return c0
    else:
        return c1 * math.cos((2.0 * j + 1.0) * i * angle_scale)

sizes = [4, 8, 16, 32]
for size in sizes:
    print(f"const float M_{size}[{size*size}] = float[](")
    vals = []
    for i in range(size):
        for j in range(size):
            vals.append(f"{get_matrix_val(i, j, size):.6f}")
    
    # Print 8 per line
    for idx in range(0, len(vals), 8):
        line = ", ".join(vals[idx:idx+8])
        if idx + 8 >= len(vals):
            print("    " + line)
        else:
            print("    " + line + ",")
    print(");")
