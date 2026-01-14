for i in range(20):
    idx = f"{i:02d}"
    print(f"""
\\begin{{figure}}[H]
    \\centering
    \\begin{{subfigure}}[b]{{0.45\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{idx}.png}}
        \\label{{fig:original_{idx}}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[b]{{0.45\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{pose_{idx}.png}}
        \\label{{fig:skeleton_{idx}}}
    \\end{{subfigure}}
\\end{{figure}}
""")